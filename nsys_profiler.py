#!/usr/bin/env python3
"""
nsys_profiler.py — NVIDIA Nsight Systems GPU Profiler for LLM Inference
========================================================================
Profiles vLLM inference with Nsight Systems (nsys) to capture:
  - CUDA kernel traces (via CUPTI)
  - cuBLAS / cuDNN call tracing
  - GPU memory copy/set operations
  - Overhead measurement (baseline vs profiled)

Architecture:
  This script uses a self-re-exec pattern to avoid subprocess path issues:
  1. OUTER mode (default): runs baseline, frees GPU, re-launches itself
     under 'nsys profile', then parses the results.
  2. INNER mode (--_inner): just loads vLLM, runs inference, saves metrics.

Usage:
  source venv/bin/activate
  python3 nsys_profiler.py --model Qwen/Qwen2.5-7B-Instruct --requests 5

Author: PInsight Benchmark Suite
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ================================================================
# COLORS
# ================================================================
C = {
    "h": "\033[1;36m",   # header cyan
    "ok": "\033[0;32m",  # green
    "w": "\033[1;33m",   # yellow warning
    "e": "\033[0;31m",   # red error
    "d": "\033[0;90m",   # dim
    "v": "\033[0;37m",   # value
    "b": "\033[1;34m",   # blue
    "r": "\033[0m",      # reset
}


def banner(text, char="═"):
    width = 70
    print(f"\n{C['h']}{char * width}")
    print(f"  {text}")
    print(f"{char * width}{C['r']}")


def section(text):
    print(f"\n{C['b']}── {text} {'─' * max(0, 60 - len(text))}{C['r']}")


def kv(key, value, unit="", indent=4):
    pad = " " * indent
    print(f"{pad}{C['d']}{key}:{C['r']} {C['v']}{value}{C['r']} {C['d']}{unit}{C['r']}")


# ================================================================
# PROMPTS (same as kineto_profiler.py for consistency)
# ================================================================

def get_prompts(n):
    base_prompts = [
        "Explain the concept of attention mechanisms in transformers in detail.",
        "What are the main differences between GPT and BERT architectures?",
        "Describe how KV-cache optimization works in autoregressive language models.",
        "Compare tensor parallelism and pipeline parallelism for distributed inference.",
        "Explain the process of tokenization and its impact on model performance.",
        "How does flash attention reduce memory usage while maintaining accuracy?",
        "Describe the role of positional encoding in transformer architectures.",
        "What optimizations does vLLM use for efficient batch inference?",
    ]
    return [base_prompts[i % len(base_prompts)] for i in range(n)]


# ================================================================
# KERNEL CATEGORIZATION (matches kineto_profiler.py)
# ================================================================

KERNEL_CATEGORIES = {
    "GEMM / MatMul": [
        "gemm", "cutlass", "cublasLt", "cublas", "matmul", "dot", "sgemm",
        "hgemm", "igemm", "wmma", "mma_", "ampere_", "sm80_", "sm90_",
    ],
    "Attention": [
        "attention", "flash_attn", "fmha", "sdpa", "flash_fwd", "flash_bwd",
        "multihead", "self_attn",
    ],
    "Normalization": [
        "layernorm", "rmsnorm", "layer_norm", "rms_norm", "batch_norm",
    ],
    "Activation": [
        "silu", "gelu", "relu", "swiglu", "sigmoid", "tanh", "activation",
    ],
    "Softmax": ["softmax", "safe_softmax"],
    "Elementwise": [
        "elementwise", "add_kernel", "mul_kernel", "fused_add",
        "binary_op", "unary_op",
    ],
    "Memory": [
        "memcpy", "memset", "copy_kernel", "fill_kernel", "scatter",
        "gather", "index_select",
    ],
    "Sampling": [
        "topk", "top_k", "top_p", "sample", "argmax", "multinomial",
    ],
    "Embedding": ["embedding", "vocab", "lookup"],
    "Communication": [
        "nccl", "all_reduce", "all_gather", "reduce_scatter", "broadcast",
    ],
    "KV Cache": ["reshape_and_cache", "paged_attention", "kv_cache"],
    "Quantization": [
        "quant", "dequant", "awq", "gptq", "int8", "fp8", "marlin",
    ],
}


def categorize_kernel(name):
    name_lower = name.lower()
    for category, keywords in KERNEL_CATEGORIES.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category
    return "Other"


# ================================================================
# NSYS SQLITE PARSER
# ================================================================

def parse_nsys_sqlite(sqlite_path):
    """Extract kernel stats from nsys SQLite export."""
    import sqlite3

    result = {}
    try:
        conn = sqlite3.connect(str(sqlite_path))

        # CUDA kernels — shortName is a FK into StringIds
        try:
            cur = conn.execute("""
                SELECT s.value, COUNT(*), SUM(k.end - k.start)
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.shortName = s.id
                GROUP BY k.shortName
                ORDER BY SUM(k.end - k.start) DESC
            """)
            kernels = {}
            total_ns = 0
            for name, count, dur_ns in cur.fetchall():
                name = str(name)
                kernels[name] = {"count": count, "total_ns": dur_ns}
                total_ns += dur_ns

            result["unique_kernels"] = len(kernels)
            result["total_kernel_launches"] = sum(v["count"] for v in kernels.values())
            result["total_cuda_time_ms"] = round(total_ns / 1e6, 1)

            # Categorize
            categories = {}
            for name, data in kernels.items():
                cat = categorize_kernel(name)
                if cat not in categories:
                    categories[cat] = {"total_ns": 0, "count": 0, "unique_kernels": 0}
                categories[cat]["total_ns"] += data["total_ns"]
                categories[cat]["count"] += data["count"]
                categories[cat]["unique_kernels"] += 1

            result["categories"] = {
                cat: {
                    "time_ms": round(d["total_ns"] / 1e6, 1),
                    "calls": d["count"],
                    "unique_kernels": d["unique_kernels"],
                    "pct": round(d["total_ns"] / total_ns * 100, 1) if total_ns > 0 else 0,
                }
                for cat, d in sorted(categories.items(),
                                     key=lambda x: x[1]["total_ns"], reverse=True)
            }

            # Top kernels
            sorted_kernels = sorted(kernels.items(),
                                    key=lambda x: x[1]["total_ns"], reverse=True)
            result["top_kernels"] = [{
                "name": name[:100],
                "category": categorize_kernel(name),
                "calls": data["count"],
                "total_ms": round(data["total_ns"] / 1e6, 1),
                "avg_us": round(data["total_ns"] / max(data["count"], 1) / 1000, 1),
                "pct": round(data["total_ns"] / total_ns * 100, 1) if total_ns > 0 else 0,
            } for name, data in sorted_kernels[:30]]

        except Exception as e:
            print(f"    {C['w']}Kernel query failed: {e}{C['r']}")

        # Memory ops
        try:
            cur = conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY")
            result["memory_copy_ops"] = cur.fetchone()[0]
        except Exception:
            result["memory_copy_ops"] = 0
        try:
            cur = conn.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMSET")
            result["memory_set_ops"] = cur.fetchone()[0]
        except Exception:
            result["memory_set_ops"] = 0

        conn.close()
    except Exception as e:
        result["parse_error"] = str(e)

    return result


def print_kernel_analysis(kernel_data):
    """Print formatted kernel analysis from parsed nsys data."""
    cats = kernel_data.get("categories", {})
    if not cats:
        print(f"    {C['w']}No CUDA kernel data extracted{C['r']}")
        return

    section("GPU Pipeline Breakdown (from nsys traces)")

    total_ms = kernel_data.get("total_cuda_time_ms", 0)
    kv("Total GPU time", f"{total_ms:.1f}", "ms")
    kv("Unique kernels", str(kernel_data.get("unique_kernels", 0)))
    kv("Total launches", f"{kernel_data.get('total_kernel_launches', 0):,}")
    kv("Memory copies", str(kernel_data.get("memory_copy_ops", 0)))
    kv("Memory sets", str(kernel_data.get("memory_set_ops", 0)))

    print(f"\n    {C['d']}{'Category':<25} {'Time (ms)':>12} {'Calls':>10} {'Kernels':>10} {'%':>8}{C['r']}")
    print(f"    {C['d']}{'─' * 68}{C['r']}")

    for cat, data in cats.items():
        print(f"    {C['v']}{cat:<25}{C['r']}"
              f" {data['time_ms']:>12.1f}"
              f" {data['calls']:>10,}"
              f" {data['unique_kernels']:>10}"
              f" {data['pct']:>7.1f}%")

    # Top kernels
    top = kernel_data.get("top_kernels", [])
    if top:
        section("Top 15 CUDA Kernels (from nsys traces)")

        print(f"\n    {C['d']}{'#':>3} {'Kernel':<50} {'Calls':>8} {'Total(ms)':>10} {'Avg(μs)':>10} {'%':>7}{C['r']}")
        print(f"    {C['d']}{'─' * 92}{C['r']}")

        for i, k in enumerate(top[:15]):
            print(f"    {C['d']}{i+1:>3}{C['r']}"
                  f" {C['v']}{k['name'][:50]:<50}{C['r']}"
                  f" {k['calls']:>8,}"
                  f" {k['total_ms']:>10.1f}"
                  f" {k['avg_us']:>10.1f}"
                  f" {k['pct']:>6.1f}%")


# ================================================================
# INNER MODE — runs under nsys, just does inference
# ================================================================

def run_inner(args):
    """Run inference and save metrics. Called when running under nsys."""
    import torch
    import torch.cuda

    print(f"=== nsys profiler inner process ===", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
    total_gb = torch.cuda.mem_get_info()[1] / (1024**3)
    print(f"GPU memory: {free_gb:.1f} / {total_gb:.1f} GB free", flush=True)

    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model} (tp={args.tensor_parallel})...", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
        max_model_len=2048,
        enforce_eager=True,
    )
    print("Model loaded.", flush=True)

    sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    all_prompts = get_prompts(args.requests)

    # Warmup
    if args.warmup > 0:
        print(f"Warmup ({args.warmup} requests)...", flush=True)
        for i in range(args.warmup):
            _ = llm.generate([all_prompts[i]], sampling_params)
        print("Warmup done.", flush=True)

    # Measured inference
    measured_prompts = all_prompts[args.warmup:]
    metrics = []

    print(f"Running {len(measured_prompts)} measured requests...", flush=True)
    for i, prompt in enumerate(measured_prompts):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        outputs = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        t1 = time.monotonic()

        n_tok = len(outputs[0].outputs[0].token_ids)
        dt = t1 - t0
        metrics.append({
            "request_index": i,
            "prompt": prompt[:80],
            "num_output_tokens": n_tok,
            "total_time_s": round(dt, 4),
            "tokens_per_second": round(n_tok / dt, 2) if dt > 0 else 0,
        })
        print(f"    [{i}] {n_tok} tok / {dt:.2f}s = {n_tok/dt:.1f} tok/s", flush=True)

    # Save metrics
    metrics_path = args._metrics_path
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}", flush=True)

    # Clean shutdown
    try:
        del llm
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass
    print("Engine shut down.", flush=True)


# ================================================================
# BASELINE (no profiling)
# ================================================================

def run_baseline(args):
    """Run inference without profiling for overhead comparison."""
    import torch
    import torch.cuda
    from vllm import LLM, SamplingParams

    banner("BASELINE — NO PROFILING")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        trust_remote_code=True,
        max_model_len=2048,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)
    all_prompts = get_prompts(args.requests)

    # Warmup
    if args.warmup > 0:
        section(f"Warmup ({args.warmup} requests)")
        for i in range(args.warmup):
            _ = llm.generate([all_prompts[i]], sampling_params)
        print(f"    {C['ok']}Warmup complete{C['r']}")

    measured_prompts = all_prompts[args.warmup:]
    metrics = []

    for i, prompt in enumerate(measured_prompts):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        outputs = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        t1 = time.monotonic()

        n_tok = len(outputs[0].outputs[0].token_ids)
        dt = t1 - t0
        metrics.append({
            "request_index": i,
            "num_output_tokens": n_tok,
            "total_time_s": round(dt, 4),
            "tokens_per_second": round(n_tok / dt, 2) if dt > 0 else 0,
        })
        print(f"    {C['v']}Request {i}: {n_tok} tokens in {dt:.2f}s "
              f"({n_tok/dt:.1f} tok/s){C['r']}")

    # Free GPU memory for the nsys pass
    print(f"    {C['d']}Freeing GPU memory for nsys pass...{C['r']}")
    del llm
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    import gc; gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
    total_mem = torch.cuda.mem_get_info()[1] / (1024**3)
    print(f"    {C['ok']}GPU memory freed: {free_mem:.1f} / {total_mem:.1f} GB available{C['r']}")

    return metrics


# ================================================================
# OVERHEAD ANALYSIS
# ================================================================

def compute_overhead(baseline_metrics, profiled_metrics):
    """Compare baseline vs profiled throughput."""
    b_tps = [m["tokens_per_second"] for m in baseline_metrics]
    p_tps = [m["tokens_per_second"] for m in profiled_metrics]

    b_mean = sum(b_tps) / len(b_tps) if b_tps else 0
    p_mean = sum(p_tps) / len(p_tps) if p_tps else 0

    overhead_pct = ((b_mean - p_mean) / b_mean * 100) if b_mean > 0 else 0

    b_times = [m["total_time_s"] for m in baseline_metrics]
    p_times = [m["total_time_s"] for m in profiled_metrics]

    return {
        "baseline_mean_tok_s": round(b_mean, 2),
        "profiled_mean_tok_s": round(p_mean, 2),
        "throughput_overhead_pct": round(overhead_pct, 2),
        "baseline_mean_time_s": round(sum(b_times) / len(b_times), 4) if b_times else 0,
        "profiled_mean_time_s": round(sum(p_times) / len(p_times), 4) if p_times else 0,
        "latency_overhead_pct": round(
            ((sum(p_times)/len(p_times) - sum(b_times)/len(b_times))
             / (sum(b_times)/len(b_times)) * 100)
            if b_times and sum(b_times) > 0 else 0, 2
        ),
    }


# ================================================================
# MAIN (OUTER MODE)
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nsight Systems GPU Profiler for vLLM Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 nsys_profiler.py --model Qwen/Qwen2.5-7B-Instruct --requests 5
  python3 nsys_profiler.py --model Qwen/Qwen2.5-7B-Instruct --requests 10 --no-overhead
  python3 nsys_profiler.py --model Qwen/Qwen2.5-7B-Instruct --output-dir ./results/nsys
        """,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--requests", type=int, default=5,
                        help="Total inference requests")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup requests (not measured)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max output tokens per request")
    parser.add_argument("--output-dir", type=str, default="./results/nsys",
                        help="Output directory")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for gated models")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Tensor parallel degree")
    parser.add_argument("--no-overhead", action="store_true",
                        help="Skip baseline overhead measurement")
    # Hidden flag: set when re-executing under nsys
    parser.add_argument("--_inner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_metrics-path", type=str, default="", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ── INNER MODE: running under nsys, just do inference ──
    if args._inner:
        run_inner(args)
        return

    # ── OUTER MODE: orchestrate baseline + nsys profiling ──

    # Check nsys is available
    nsys_bin = shutil.which("nsys")
    if not nsys_bin:
        print(f"{C['e']}Error: nsys not found in PATH.{C['r']}")
        print(f"{C['d']}Install NVIDIA Nsight Systems or add it to PATH.{C['r']}")
        sys.exit(1)

    # Get nsys version
    try:
        ver = subprocess.run([nsys_bin, "--version"], capture_output=True, text=True)
        nsys_version = ver.stdout.strip() if ver.returncode == 0 else "unknown"
    except Exception:
        nsys_version = "unknown"

    # Setup output directories (ABSOLUTE paths to avoid cwd issues)
    model_safe = args.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (Path(args.output_dir) / f"{model_safe}_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nsys_traces_dir = output_dir / "nsys_traces"
    nsys_traces_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    import torch
    import torch.cuda

    banner("NSIGHT SYSTEMS GPU PROFILER (vLLM)", "▓")
    print(f"\n    {C['d']}Model:           {args.model}{C['r']}")
    print(f"    {C['d']}Tensor Parallel: {args.tensor_parallel}{C['r']}")
    print(f"    {C['d']}Requests:        {args.requests} (warmup: {args.warmup}){C['r']}")
    print(f"    {C['d']}Max Tokens:      {args.max_tokens}{C['r']}")
    print(f"    {C['d']}Output:          {output_dir}{C['r']}")
    print(f"    {C['d']}nsys:            {nsys_version}{C['r']}")
    print(f"    {C['d']}PyTorch:         {torch.__version__}{C['r']}")
    print(f"    {C['d']}CUDA:            {torch.version.cuda}{C['r']}")
    print(f"    {C['d']}GPUs:            {torch.cuda.device_count()}× {torch.cuda.get_device_name(0)}{C['r']}")

    # ── Optional baseline ──
    baseline_metrics = None
    overhead_data = None
    if not args.no_overhead:
        baseline_metrics = run_baseline(args)

    # ── Launch nsys profiling ──
    banner("NSYS PROFILER — ACTIVE")

    report_path = nsys_traces_dir / "nsys_trace"
    metrics_path = output_dir / "_nsys_inner_metrics.json"

    print(f"    {C['d']}Report:  {report_path}.nsys-rep{C['r']}")
    print(f"    {C['d']}Metrics: {metrics_path}{C['r']}")
    print(f"    {C['d']}Launching self under nsys profile...{C['r']}")

    # Re-exec this same script under nsys with --_inner flag
    python_bin = sys.executable
    this_script = str(Path(__file__).resolve())

    inner_cmd = [
        python_bin, this_script,
        "--_inner",
        "--model", args.model,
        "--requests", str(args.requests),
        "--warmup", str(args.warmup),
        "--max-tokens", str(args.max_tokens),
        "--tensor-parallel", str(args.tensor_parallel),
        "--_metrics-path", str(metrics_path),
    ]
    if args.hf_token:
        inner_cmd.extend(["--hf-token", args.hf_token])

    nsys_cmd = [
        nsys_bin, "profile",
        "--trace", "cuda,nvtx,cublas",
        "--cuda-memory-usage", "true",
        "--output", str(report_path),
        "--force-overwrite", "true",
        "--kill", "sigterm",
        "--",
    ] + inner_cmd

    nsys_start = time.monotonic()
    try:
        result = subprocess.run(
            nsys_cmd,
            timeout=600,
            cwd=str(output_dir),
        )
        nsys_elapsed = time.monotonic() - nsys_start

        if result.returncode != 0:
            print(f"    {C['w']}nsys exited with code {result.returncode}{C['r']}")
            rep = Path(f"{report_path}.nsys-rep")
            if rep.exists():
                print(f"    {C['ok']}Trace file exists despite non-zero exit — continuing{C['r']}")
            else:
                print(f"    {C['e']}No trace file produced.{C['r']}")
        else:
            print(f"    {C['ok']}nsys completed in {nsys_elapsed:.1f}s{C['r']}")

    except subprocess.TimeoutExpired:
        print(f"    {C['e']}nsys timed out after 600s{C['r']}")
        nsys_elapsed = 600

    # ── Read profiled metrics ──
    profiled_metrics = []
    if metrics_path.exists():
        with open(metrics_path) as f:
            profiled_metrics = json.load(f)
        print(f"    {C['ok']}Loaded {len(profiled_metrics)} profiled metric entries{C['r']}")
    else:
        print(f"    {C['w']}No metrics file — inner process may have crashed{C['r']}")

    # ── Parse nsys traces ──
    section("Analyzing nsys Trace Files")

    nsys_data = {
        "nsys_version": nsys_version,
        "total_nsys_time_s": round(nsys_elapsed, 1),
    }

    rep_file = Path(f"{report_path}.nsys-rep")
    if rep_file.exists():
        nsys_data["report_file"] = str(rep_file)
        nsys_data["report_size_mb"] = round(rep_file.stat().st_size / (1024 * 1024), 1)
        kv("Report file", rep_file.name)
        kv("Report size", f"{nsys_data['report_size_mb']:.1f}", "MB")

        # Export to SQLite and parse kernel data
        sqlite_path = nsys_traces_dir / "nsys_trace.sqlite"
        try:
            print(f"    {C['d']}Exporting to SQLite...{C['r']}")
            exp = subprocess.run(
                [nsys_bin, "export", "--type", "sqlite",
                 "--output", str(sqlite_path), str(rep_file)],
                capture_output=True, timeout=120,
            )
            if sqlite_path.exists():
                kernel_data = parse_nsys_sqlite(sqlite_path)
                nsys_data.update(kernel_data)
                print_kernel_analysis(kernel_data)
            else:
                print(f"    {C['w']}SQLite export produced no file{C['r']}")
        except subprocess.TimeoutExpired:
            print(f"    {C['w']}SQLite export timed out{C['r']}")
        except Exception as e:
            print(f"    {C['w']}SQLite export failed: {e}{C['r']}")

        print(f"\n    {C['d']}View traces with: nsys-ui {rep_file}{C['r']}")
    else:
        print(f"    {C['w']}No .nsys-rep file found{C['r']}")

    # ── Overhead analysis ──
    if baseline_metrics and profiled_metrics:
        section("Overhead Analysis")
        overhead_data = compute_overhead(baseline_metrics, profiled_metrics)

        print(f"\n    {C['d']}{'Metric':<30} {'Baseline':>14} {'nsys':>14} {'Overhead':>10}{C['r']}")
        print(f"    {C['d']}{'─' * 72}{C['r']}")
        print(f"    {C['v']}{'Mean Throughput (tok/s)':<30}{C['r']}"
              f" {overhead_data['baseline_mean_tok_s']:>14.1f}"
              f" {overhead_data['profiled_mean_tok_s']:>14.1f}"
              f" {C['w']}{overhead_data['throughput_overhead_pct']:>+9.1f}%{C['r']}")
        print(f"    {C['v']}{'Mean Latency (s/req)':<30}{C['r']}"
              f" {overhead_data['baseline_mean_time_s']:>14.3f}"
              f" {overhead_data['profiled_mean_time_s']:>14.3f}"
              f" {C['w']}{overhead_data['latency_overhead_pct']:>+9.1f}%{C['r']}")

    # ── Save final results ──
    section("Saving Results")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "backend": "vllm",
        "profiler": "nsight_systems",
        "nsys_version": nsys_version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "tensor_parallel": args.tensor_parallel,
        "max_tokens": args.max_tokens,
        "total_requests": args.requests,
        "warmup_requests": args.warmup,
        "inference_metrics": profiled_metrics,
        "profiler_data": nsys_data,
        "overhead": overhead_data,
        "aggregate": {
            "mean_tok_s": round(
                sum(m["tokens_per_second"] for m in profiled_metrics)
                / len(profiled_metrics), 2
            ) if profiled_metrics else 0,
            "total_tokens": sum(m["num_output_tokens"] for m in profiled_metrics),
            "mean_time_s": round(
                sum(m["total_time_s"] for m in profiled_metrics)
                / len(profiled_metrics), 3
            ) if profiled_metrics else 0,
        },
        "capabilities": {
            "cuda_kernel_tracing": True,
            "cpu_operator_tracing": True,
            "memory_profiling": True,
            "memory_bandwidth_analysis": True,
            "pcie_transfer_tracking": True,
            "nvlink_traffic": True,
            "system_wide_profiling": True,
            "multi_process_support": True,
            "operator_kernel_correlation": False,
            "python_stack_traces": False,
            "chrome_trace_export": False,
            "requires_external_tool": True,
            "requires_root": False,
            "embeddable_in_python": False,
        },
    }

    results_path = output_dir / "nsys_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    kv("Results JSON", str(results_path))

    # Clean up temp metrics file
    if metrics_path.exists():
        metrics_path.unlink()

    # ── Summary ──
    banner("NSYS PROFILING COMPLETE", "▓")
    print(f"\n    {C['d']}Results:       {results_path}{C['r']}")
    if rep_file.exists():
        print(f"    {C['d']}Trace file:    {rep_file}{C['r']}")
        print(f"    {C['d']}Trace size:    {nsys_data.get('report_size_mb', 0):.1f} MB{C['r']}")
    agg = results["aggregate"]
    print(f"    {C['d']}Mean tok/s:    {agg['mean_tok_s']}{C['r']}")
    print(f"    {C['d']}Total tokens:  {agg['total_tokens']}{C['r']}")
    if overhead_data:
        print(f"    {C['d']}Overhead:      {overhead_data['throughput_overhead_pct']:+.1f}% throughput, "
              f"{overhead_data['latency_overhead_pct']:+.1f}% latency{C['r']}")
    if nsys_data.get("total_cuda_time_ms"):
        print(f"    {C['d']}GPU time:      {nsys_data['total_cuda_time_ms']:.1f} ms across "
              f"{nsys_data.get('total_kernel_launches', 0):,} kernel launches{C['r']}")
    print()


if __name__ == "__main__":
    main()
