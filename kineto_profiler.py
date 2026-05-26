#!/usr/bin/env python3
"""
kineto_profiler.py — PyTorch Kineto GPU Profiler for LLM Inference
==================================================================
Uses torch.profiler (Kineto backend) to profile vLLM inference with:
  - CUDA kernel tracing via CUPTI
  - CPU operator-level breakdown
  - GPU memory allocation tracking
  - Chrome trace export (viewable in chrome://tracing or perfetto.dev)
  - Overhead measurement (with vs without profiling)

Usage:
  source venv/bin/activate
  python3 kineto_profiler.py --model Qwen/Qwen2.5-7B-Instruct --requests 5

Author: PInsight Benchmark Suite
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.cuda
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)


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
# PROMPTS (same as vllm_profile.py for consistency)
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
# KERNEL CATEGORIZATION (matches trace_analyzer.py categories)
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
        "group_norm",
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
        "categorical",
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
# PYTORCH VERSION COMPAT
# ================================================================
# PyTorch 2.9+ renamed profiler event attributes:
#   self_cuda_time_total  -> self_device_time_total
#   cuda_time_total       -> device_time_total
#   cuda_memory_usage     -> device_memory_usage
#   self_cuda_memory_usage -> self_device_memory_usage
# This helper tries the new name first, falls back to old.

def _evt_attr(evt, attr_name, default=0):
    """Get profiler event attribute with cuda/device name compat."""
    # Try the name as given
    if hasattr(evt, attr_name):
        return getattr(evt, attr_name)
    # Try swapping cuda <-> device
    if "cuda" in attr_name:
        alt = attr_name.replace("cuda", "device")
    elif "device" in attr_name:
        alt = attr_name.replace("device", "cuda")
    else:
        return default
    if hasattr(evt, alt):
        return getattr(evt, alt)
    return default


def _sort_key_compat():
    """Return the correct sort key for key_averages().table()."""
    # PyTorch 2.9+ uses 'self_device_time_total'
    try:
        from torch.autograd.profiler_util import EventList
        # Quick probe: create a dummy to check accepted keys
        return "self_device_time_total"
    except Exception:
        return "self_cuda_time_total"


# ================================================================
# KINETO PROFILER (via vLLM's built-in profiling hooks)
# ================================================================

def run_kineto_profile(llm, prompts, sampling_params, output_dir, args):
    """
    Run inference with vLLM's built-in Kineto profiling.

    Uses llm.start_profile() / llm.stop_profile() which instruments the
    EngineCore subprocess where GPU kernels actually execute. The env var
    VLLM_TORCH_PROFILER_DIR must be set BEFORE LLM init.
    """

    banner("KINETO PROFILER — ACTIVE (vLLM EngineCore)")
    trace_dir = output_dir / "kineto_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    print(f"    {C['d']}Mode: vLLM start_profile/stop_profile{C['r']}")
    print(f"    {C['d']}Trace dir: {os.environ.get('VLLM_TORCH_PROFILER_DIR', 'NOT SET')}{C['r']}")
    print(f"    {C['d']}Requests: {len(prompts)}{C['r']}")

    metrics = []

    # Start vLLM's internal profiler (captures GPU kernels in EngineCore)
    llm.start_profile()

    for i, prompt in enumerate(prompts):
        t_start = time.monotonic()
        outputs = llm.generate([prompt], sampling_params)
        t_end = time.monotonic()

        output = outputs[0]
        num_tokens = len(output.outputs[0].token_ids)
        total_s = t_end - t_start

        metrics.append({
            "request_index": i,
            "prompt": prompt[:80],
            "num_output_tokens": num_tokens,
            "total_time_s": round(total_s, 4),
            "tokens_per_second": round(num_tokens / total_s, 2) if total_s > 0 else 0,
        })
        print(f"    {C['ok']}Request {i}: {num_tokens} tokens in {total_s:.2f}s "
              f"({num_tokens/total_s:.1f} tok/s){C['r']}")

    # Stop profiler — this triggers trace file export
    llm.stop_profile()
    print(f"    {C['ok']}Profiler stopped, traces exported{C['r']}")

    # ── Find and analyze trace files ──
    section("Analyzing vLLM Trace Files")

    profiler_dir = Path(os.environ.get("VLLM_TORCH_PROFILER_DIR", str(trace_dir)))
    trace_files = sorted(profiler_dir.rglob("*.json"))

    if not trace_files:
        # Also check the trace_dir itself
        trace_files = sorted(trace_dir.rglob("*.json"))

    profiler_results = {
        "tool": "kineto_vllm",
        "method": "start_profile/stop_profile",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    if trace_files:
        print(f"    {C['ok']}Found {len(trace_files)} trace file(s):{C['r']}")
        total_trace_size = 0
        for tf in trace_files:
            size_mb = tf.stat().st_size / (1024 * 1024)
            total_trace_size += size_mb
            print(f"      {C['v']}{tf.name} ({size_mb:.1f} MB){C['r']}")

        profiler_results["trace_files"] = [str(f) for f in trace_files]
        profiler_results["total_trace_size_mb"] = round(total_trace_size, 1)

        # Parse the Chrome trace JSON for kernel data
        kernel_data = _parse_chrome_traces(trace_files)
        profiler_results.update(kernel_data)

        # Print kernel analysis
        _print_kernel_analysis(kernel_data)

        print(f"\n    {C['d']}View traces in: chrome://tracing or https://ui.perfetto.dev{C['r']}")
    else:
        print(f"    {C['w']}No trace files found in {profiler_dir}{C['r']}")
        print(f"    {C['w']}Checked: {profiler_dir}{C['r']}")

    return metrics, profiler_results


def _parse_chrome_traces(trace_files):
    """Parse Chrome trace JSON files to extract CUDA kernel data."""
    all_kernels = {}  # name -> {total_us, count}
    categories = {}
    total_cuda_us = 0
    memory_events = 0

    for trace_file in trace_files:
        try:
            with open(trace_file) as f:
                trace = json.load(f)

            events = trace if isinstance(trace, list) else trace.get("traceEvents", [])

            for evt in events:
                if not isinstance(evt, dict):
                    continue

                cat = evt.get("cat", "")
                name = evt.get("name", "")
                dur = evt.get("dur", 0)  # microseconds

                # CUDA kernels show up as "kernel" or "gpu_memcpy" category
                if cat in ("kernel", "cuda_runtime", "gpu_memcpy", "gpu_memset"):
                    if name not in all_kernels:
                        all_kernels[name] = {"total_us": 0, "count": 0}
                    all_kernels[name]["total_us"] += dur
                    all_kernels[name]["count"] += 1
                    total_cuda_us += dur

                    k_cat = categorize_kernel(name)
                    if k_cat not in categories:
                        categories[k_cat] = {"total_us": 0, "count": 0, "kernels": set()}
                    categories[k_cat]["total_us"] += dur
                    categories[k_cat]["count"] += 1
                    categories[k_cat]["kernels"].add(name)

                if cat in ("gpu_memcpy", "gpu_memset"):
                    memory_events += 1

        except Exception as e:
            print(f"    {C['w']}Error parsing {trace_file.name}: {e}{C['r']}")

    # Sort kernels by total time
    sorted_kernels = sorted(all_kernels.items(), key=lambda x: x[1]["total_us"], reverse=True)

    # Convert categories (sets not serializable)
    cat_data = {}
    for cat, data in sorted(categories.items(), key=lambda x: x[1]["total_us"], reverse=True):
        cat_data[cat] = {
            "time_ms": round(data["total_us"] / 1000, 1),
            "calls": data["count"],
            "unique_kernels": len(data["kernels"]),
            "pct": round(data["total_us"] / total_cuda_us * 100, 1) if total_cuda_us > 0 else 0,
        }

    return {
        "total_cuda_time_ms": round(total_cuda_us / 1000, 1),
        "unique_kernels": len(all_kernels),
        "total_kernel_launches": sum(v["count"] for v in all_kernels.values()),
        "memory_events": memory_events,
        "categories": cat_data,
        "top_kernels": [{
            "name": name[:100],
            "category": categorize_kernel(name),
            "calls": data["count"],
            "total_ms": round(data["total_us"] / 1000, 1),
            "avg_us": round(data["total_us"] / max(data["count"], 1), 1),
            "pct": round(data["total_us"] / total_cuda_us * 100, 1) if total_cuda_us > 0 else 0,
        } for name, data in sorted_kernels[:30]],
    }


def _print_kernel_analysis(kernel_data):
    """Print formatted kernel analysis from parsed trace data."""

    cats = kernel_data.get("categories", {})
    if not cats:
        print(f"    {C['w']}No CUDA kernels found in traces{C['r']}")
        return

    section("GPU Pipeline Breakdown (from vLLM engine traces)")

    total_ms = kernel_data.get("total_cuda_time_ms", 0)
    kv("Total GPU time", f"{total_ms:.1f}", "ms")
    kv("Unique kernels", str(kernel_data.get("unique_kernels", 0)))
    kv("Total launches", f"{kernel_data.get('total_kernel_launches', 0):,}")
    kv("Memory events", str(kernel_data.get("memory_events", 0)))

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
        section("Top 15 CUDA Kernels (from vLLM engine traces)")

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
# BASELINE (no profiling)
# ================================================================

def run_baseline(llm, prompts, sampling_params):
    """Run inference without any profiling for overhead comparison."""
    banner("BASELINE — NO PROFILING")

    metrics = []
    for i, prompt in enumerate(prompts):
        t_start = time.monotonic()
        outputs = llm.generate([prompt], sampling_params)
        t_end = time.monotonic()

        output = outputs[0]
        num_tokens = len(output.outputs[0].token_ids)
        total_s = t_end - t_start

        metrics.append({
            "request_index": i,
            "num_output_tokens": num_tokens,
            "total_time_s": round(total_s, 4),
            "tokens_per_second": round(num_tokens / total_s, 2) if total_s > 0 else 0,
        })
        print(f"    {C['v']}Request {i}: {num_tokens} tokens in {total_s:.2f}s "
              f"({num_tokens/total_s:.1f} tok/s){C['r']}")

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
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kineto GPU Profiler for vLLM Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 kineto_profiler.py --model Qwen/Qwen2.5-7B-Instruct --requests 5
  python3 kineto_profiler.py --model Qwen/Qwen2.5-7B-Instruct --requests 10 --no-overhead
  python3 kineto_profiler.py --model Qwen/Qwen2.5-7B-Instruct --output-dir ./results/kineto
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
    parser.add_argument("--output-dir", type=str, default="./results/kineto",
                        help="Output directory")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for gated models")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Tensor parallel degree")
    parser.add_argument("--no-overhead", action="store_true",
                        help="Skip baseline overhead measurement")

    args = parser.parse_args()

    # ── Set VLLM_TORCH_PROFILER_DIR BEFORE importing/creating LLM ──
    # This tells vLLM's EngineCore to enable Kineto tracing inside the
    # worker subprocess where GPU kernels actually execute.
    model_safe = args.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (Path(args.output_dir) / f"{model_safe}_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = output_dir / "kineto_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str(trace_dir)

    from vllm import LLM, SamplingParams

    banner("KINETO GPU PROFILER (vLLM Engine)", "▓")
    print(f"\n    {C['d']}Model:           {args.model}{C['r']}")
    print(f"    {C['d']}Tensor Parallel: {args.tensor_parallel}{C['r']}")
    print(f"    {C['d']}Requests:        {args.requests} (warmup: {args.warmup}){C['r']}")
    print(f"    {C['d']}Max Tokens:      {args.max_tokens}{C['r']}")
    print(f"    {C['d']}Output:          {output_dir}{C['r']}")
    print(f"    {C['d']}Trace Dir:       {trace_dir}{C['r']}")
    print(f"    {C['d']}PyTorch:         {torch.__version__}{C['r']}")
    print(f"    {C['d']}CUDA:            {torch.version.cuda}{C['r']}")
    print(f"    {C['d']}GPUs:            {torch.cuda.device_count()}× {torch.cuda.get_device_name(0)}{C['r']}")
    print(f"    {C['d']}Profiling via:   VLLM_TORCH_PROFILER_DIR + start_profile/stop_profile{C['r']}")

    # ── Load model ──
    section("Loading Model")
    load_start = time.monotonic()

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel,
        "trust_remote_code": True,
        "max_model_len": 2048,
        "enforce_eager": True,  # disable CUDA graphs for cleaner profiling
    }
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    llm = LLM(**llm_kwargs)
    load_elapsed = time.monotonic() - load_start
    kv("Model loaded in", f"{load_elapsed:.1f}", "s")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=args.max_tokens,
    )

    all_prompts = get_prompts(args.requests)

    # ── Warmup ──
    if args.warmup > 0:
        section(f"Warmup ({args.warmup} requests)")
        for i in range(args.warmup):
            _ = llm.generate([all_prompts[i]], sampling_params)
        print(f"    {C['ok']}Warmup complete{C['r']}")

    measured_prompts = all_prompts[args.warmup:]

    # ── Baseline ──
    overhead_data = None
    if not args.no_overhead:
        baseline_metrics = run_baseline(llm, measured_prompts, sampling_params)
    else:
        baseline_metrics = None

    # ── Kineto profiling (via vLLM's engine hooks) ──
    profiled_metrics, profiler_results = run_kineto_profile(
        llm, measured_prompts, sampling_params, output_dir, args
    )

    # ── Overhead analysis ──
    if baseline_metrics is not None:
        section("Overhead Analysis")
        overhead_data = compute_overhead(baseline_metrics, profiled_metrics)

        print(f"\n    {C['d']}{'Metric':<30} {'Baseline':>14} {'Kineto':>14} {'Overhead':>10}{C['r']}")
        print(f"    {C['d']}{'─' * 72}{C['r']}")
        print(f"    {C['v']}{'Mean Throughput (tok/s)':<30}{C['r']}"
              f" {overhead_data['baseline_mean_tok_s']:>14.1f}"
              f" {overhead_data['profiled_mean_tok_s']:>14.1f}"
              f" {C['w']}{overhead_data['throughput_overhead_pct']:>+9.1f}%{C['r']}")
        print(f"    {C['v']}{'Mean Latency (s/req)':<30}{C['r']}"
              f" {overhead_data['baseline_mean_time_s']:>14.3f}"
              f" {overhead_data['profiled_mean_time_s']:>14.3f}"
              f" {C['w']}{overhead_data['latency_overhead_pct']:>+9.1f}%{C['r']}")

    # ── Save results ──
    section("Saving Results")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "backend": "vllm",
        "profiler": "kineto_vllm",
        "method": "VLLM_TORCH_PROFILER_DIR + start_profile/stop_profile",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "tensor_parallel": args.tensor_parallel,
        "max_tokens": args.max_tokens,
        "total_requests": args.requests,
        "warmup_requests": args.warmup,
        "model_load_time_s": round(load_elapsed, 2),
        "inference_metrics": profiled_metrics,
        "profiler_data": profiler_results,
        "overhead": overhead_data,
        "aggregate": {
            "mean_tok_s": round(
                sum(m["tokens_per_second"] for m in profiled_metrics)
                / len(profiled_metrics), 2
            ),
            "total_tokens": sum(m["num_output_tokens"] for m in profiled_metrics),
            "mean_time_s": round(
                sum(m["total_time_s"] for m in profiled_metrics)
                / len(profiled_metrics), 3
            ),
        },
        "capabilities": {
            "cuda_kernel_tracing": True,
            "cpu_operator_tracing": True,
            "memory_profiling": True,
            "operator_kernel_correlation": True,
            "chrome_trace_export": True,
            "captures_engine_subprocess": True,
            "requires_external_tool": False,
            "requires_root": False,
            "embeddable_in_python": True,
        },
    }

    results_path = output_dir / "kineto_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    kv("Results JSON", str(results_path))

    # ── Summary ──
    banner("KINETO PROFILING COMPLETE", "▓")
    print(f"\n    {C['d']}Results:       {results_path}{C['r']}")
    trace_files = profiler_results.get("trace_files", [])
    if trace_files:
        print(f"    {C['d']}Trace files:   {len(trace_files)} file(s) in {trace_dir}{C['r']}")
        print(f"    {C['d']}Trace size:    {profiler_results.get('total_trace_size_mb', 0):.1f} MB{C['r']}")
    agg = results["aggregate"]
    print(f"    {C['d']}Mean tok/s:    {agg['mean_tok_s']}{C['r']}")
    print(f"    {C['d']}Total tokens:  {agg['total_tokens']}{C['r']}")
    if overhead_data:
        print(f"    {C['d']}Overhead:      {overhead_data['throughput_overhead_pct']:+.1f}% throughput, "
              f"{overhead_data['latency_overhead_pct']:+.1f}% latency{C['r']}")
    kdata = profiler_results.get("total_cuda_time_ms", 0)
    if kdata:
        print(f"    {C['d']}GPU time:      {kdata:.1f} ms across "
              f"{profiler_results.get('total_kernel_launches', 0):,} kernel launches{C['r']}")
    print()


if __name__ == "__main__":
    main()
