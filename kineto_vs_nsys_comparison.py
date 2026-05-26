#!/usr/bin/env python3
"""
kineto_vs_nsys_comparison.py — Head-to-Head Profiler Overhead Comparison
========================================================================
Runs the same vLLM inference workload under three conditions:
  1. Baseline (no profiling)
  2. PyTorch Kineto (torch.profiler)
  3. NVIDIA Nsight Systems (nsys subprocess)

Measures throughput impact, latency overhead, and compares captured data.
Generates a structured JSON report + terminal comparison table.

Usage:
  source venv/bin/activate
  python3 kineto_vs_nsys_comparison.py --model Qwen/Qwen2.5-7B-Instruct --requests 5

Author: PInsight Benchmark Suite
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.cuda
from torch.profiler import ProfilerActivity, profile, record_function


# ================================================================
# PYTORCH VERSION COMPAT
# ================================================================
def _evt_attr(evt, attr_name, default=0):
    """Get profiler event attribute with cuda/device name compat."""
    if hasattr(evt, attr_name):
        return getattr(evt, attr_name)
    if "cuda" in attr_name:
        alt = attr_name.replace("cuda", "device")
    elif "device" in attr_name:
        alt = attr_name.replace("device", "cuda")
    else:
        return default
    if hasattr(evt, alt):
        return getattr(evt, alt)
    return default


# ================================================================
# COLORS & FORMATTING
# ================================================================
C = {
    "h": "\033[1;36m", "ok": "\033[0;32m", "w": "\033[1;33m",
    "e": "\033[0;31m", "d": "\033[0;90m", "v": "\033[0;37m",
    "b": "\033[1;34m", "m": "\033[1;35m", "r": "\033[0m",
}

def banner(text, char="═"):
    w = 76
    print(f"\n{C['h']}{char * w}")
    print(f"  {text}")
    print(f"{char * w}{C['r']}")

def section(text):
    print(f"\n{C['b']}── {text} {'─' * max(0, 64 - len(text))}{C['r']}")

def kv(key, value, unit="", indent=4):
    pad = " " * indent
    print(f"{pad}{C['d']}{key}:{C['r']} {C['v']}{value}{C['r']} {C['d']}{unit}{C['r']}")


# ================================================================
# PROMPTS
# ================================================================

def get_prompts(n):
    base = [
        "Explain the concept of attention mechanisms in transformers in detail.",
        "What are the main differences between GPT and BERT architectures?",
        "Describe how KV-cache optimization works in autoregressive language models.",
        "Compare tensor parallelism and pipeline parallelism for distributed inference.",
        "Explain the process of tokenization and its impact on model performance.",
        "How does flash attention reduce memory usage while maintaining accuracy?",
        "Describe the role of positional encoding in transformer architectures.",
        "What optimizations does vLLM use for efficient batch inference?",
    ]
    return [base[i % len(base)] for i in range(n)]


# ================================================================
# PASS 1: BASELINE (no profiling)
# ================================================================

def run_baseline(llm, prompts, sampling_params):
    """Measure raw throughput with no profiling overhead."""
    banner("PASS 1/3 — BASELINE (no profiling)")

    metrics = []
    total_start = time.monotonic()

    for i, prompt in enumerate(prompts):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        outputs = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        t1 = time.monotonic()

        n_tok = len(outputs[0].outputs[0].token_ids)
        dt = t1 - t0
        metrics.append({
            "request": i, "tokens": n_tok,
            "time_s": round(dt, 4),
            "tok_s": round(n_tok / dt, 2) if dt > 0 else 0,
        })
        print(f"    {C['v']}[{i}] {n_tok} tok / {dt:.2f}s = {n_tok/dt:.1f} tok/s{C['r']}")

    total_time = time.monotonic() - total_start
    tps_values = [m["tok_s"] for m in metrics]

    return {
        "pass": "baseline",
        "total_time_s": round(total_time, 3),
        "mean_tok_s": round(sum(tps_values) / len(tps_values), 2),
        "min_tok_s": round(min(tps_values), 2),
        "max_tok_s": round(max(tps_values), 2),
        "total_tokens": sum(m["tokens"] for m in metrics),
        "per_request": metrics,
    }


# ================================================================
# PASS 2: KINETO (torch.profiler)
# ================================================================

def run_kineto(llm, prompts, sampling_params, output_dir):
    """Profile with vLLM's built-in Kineto (start_profile / stop_profile)."""
    banner("PASS 2/3 — KINETO (vLLM engine profiler)")

    trace_dir = output_dir / "kineto_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    total_start = time.monotonic()

    # Start vLLM's internal profiler (captures GPU kernels in EngineCore)
    llm.start_profile()

    for i, prompt in enumerate(prompts):
        t0 = time.monotonic()
        outputs = llm.generate([prompt], sampling_params)
        t1 = time.monotonic()

        n_tok = len(outputs[0].outputs[0].token_ids)
        dt = t1 - t0
        metrics.append({
            "request": i, "tokens": n_tok,
            "time_s": round(dt, 4),
            "tok_s": round(n_tok / dt, 2) if dt > 0 else 0,
        })
        print(f"    {C['v']}[{i}] {n_tok} tok / {dt:.2f}s = {n_tok/dt:.1f} tok/s{C['r']}")

    # Stop profiler — triggers trace export
    llm.stop_profile()
    total_time = time.monotonic() - total_start
    print(f"    {C['ok']}Profiler stopped, traces exported{C['r']}")

    # Find and parse trace files
    profiler_dir = Path(os.environ.get("VLLM_TORCH_PROFILER_DIR", str(trace_dir)))
    trace_files = sorted(profiler_dir.rglob("*.json"))
    trace_size_mb = sum(f.stat().st_size / (1024 * 1024) for f in trace_files)

    # Parse kernel data from Chrome traces
    kernel_count = 0
    unique_kernels = set()
    category_times = {}
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
                dur = evt.get("dur", 0)
                if cat in ("kernel", "cuda_runtime", "gpu_memcpy", "gpu_memset"):
                    unique_kernels.add(name)
                    kernel_count += 1
                    total_cuda_us += dur
                    k_cat = _categorize(name)
                    if k_cat not in category_times:
                        category_times[k_cat] = 0
                    category_times[k_cat] += dur
                if cat in ("gpu_memcpy", "gpu_memset"):
                    memory_events += 1
        except Exception as e:
            print(f"    {C['w']}Error parsing {trace_file.name}: {e}{C['r']}")

    if trace_files:
        print(f"    {C['ok']}Parsed {len(trace_files)} trace file(s): "
              f"{len(unique_kernels)} unique kernels, {kernel_count:,} launches{C['r']}")

    tps_values = [m["tok_s"] for m in metrics]

    return {
        "pass": "kineto",
        "total_time_s": round(total_time, 3),
        "mean_tok_s": round(sum(tps_values) / len(tps_values), 2),
        "min_tok_s": round(min(tps_values), 2),
        "max_tok_s": round(max(tps_values), 2),
        "total_tokens": sum(m["tokens"] for m in metrics),
        "per_request": metrics,
        "captured_data": {
            "unique_kernels": len(unique_kernels),
            "total_kernel_launches": kernel_count,
            "total_cuda_time_ms": round(total_cuda_us / 1000, 1),
            "categories": {cat: round(us / 1000, 1)
                           for cat, us in sorted(category_times.items(),
                                                 key=lambda x: x[1], reverse=True)},
            "memory_events": memory_events,
            "trace_files": [str(f) for f in trace_files],
            "trace_size_mb": round(trace_size_mb, 1),
        },
    }


# ================================================================
# PASS 3: NSYS (external Nsight Systems)
# ================================================================

def run_nsys(llm, prompts, sampling_params, output_dir, args):
    """
    Measure throughput with nsys wrapping the process.
    Since nsys must wrap the entire process, we measure its overhead by
    running a subprocess with nsys profile.
    """
    banner("PASS 3/3 — NSIGHT SYSTEMS (nsys)")

    nsys_bin = shutil.which("nsys")
    if not nsys_bin:
        print(f"    {C['e']}nsys not found in PATH. Skipping nsys pass.{C['r']}")
        return None

    # Get nsys version
    try:
        ver = subprocess.run([nsys_bin, "--version"], capture_output=True, text=True)
        nsys_version = ver.stdout.strip() if ver.returncode == 0 else "unknown"
    except Exception:
        nsys_version = "unknown"
    kv("nsys version", nsys_version)

    # Write a small runner script that nsys will profile
    nsys_dir = output_dir / "nsys_traces"
    nsys_dir.mkdir(parents=True, exist_ok=True)

    runner_script = output_dir / "_nsys_runner.py"
    runner_script.write_text(f'''#!/usr/bin/env python3
"""Auto-generated nsys runner for overhead comparison."""
import json, time, sys, os
import torch, torch.cuda

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from vllm import LLM, SamplingParams

model = "{args.model}"
tp = {args.tensor_parallel}
max_tokens = {args.max_tokens}
warmup = {args.warmup}

prompts = {json.dumps(get_prompts(args.requests))}

llm = LLM(
    model=model,
    tensor_parallel_size=tp,
    trust_remote_code=True,
    max_model_len=2048,
    enforce_eager=True,
)
sp = SamplingParams(temperature=0.7, max_tokens=max_tokens)

# Warmup
for i in range(warmup):
    _ = llm.generate([prompts[i]], sp)

# Measured
metrics = []
measured = prompts[warmup:]
for i, p in enumerate(measured):
    torch.cuda.synchronize()
    t0 = time.monotonic()
    out = llm.generate([p], sp)
    torch.cuda.synchronize()
    t1 = time.monotonic()
    n = len(out[0].outputs[0].token_ids)
    dt = t1 - t0
    metrics.append({{"request": i, "tokens": n, "time_s": round(dt, 4),
                     "tok_s": round(n/dt, 2) if dt > 0 else 0}})
    print(f"    [{{i}}] {{n}} tok / {{dt:.2f}}s = {{n/dt:.1f}} tok/s")

# Save metrics
out_path = "{nsys_dir}/nsys_metrics.json"
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved: {{out_path}}")
''')

    report_name = nsys_dir / "nsys_comparison"

    print(f"    {C['d']}Launching nsys profile subprocess...{C['r']}")
    print(f"    {C['d']}This will load the model again inside nsys.{C['r']}")
    print(f"    {C['d']}Report: {report_name}{C['r']}")

    # Determine active venv python
    python_bin = sys.executable

    cmd = [
        nsys_bin, "profile",
        "--trace-fork-before-exec=true",
        "--output", str(report_name),
        "--trace", "cuda,nvtx,cublas",
        "--cuda-memory-usage", "true",
        "--force-overwrite", "true",
        "--stats", "true",
        "--",
        python_bin, str(runner_script),
    ]

    nsys_start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout
            cwd=str(output_dir.parent),
        )
        nsys_elapsed = time.monotonic() - nsys_start

        if result.returncode != 0:
            print(f"    {C['e']}nsys exited with code {result.returncode}{C['r']}")
            if result.stderr:
                for line in result.stderr.split('\n')[:10]:
                    print(f"    {C['e']}  {line}{C['r']}")

    except subprocess.TimeoutExpired:
        print(f"    {C['e']}nsys timed out after 600s{C['r']}")
        return None

    # Read metrics from the subprocess
    nsys_metrics_path = nsys_dir / "nsys_metrics.json"
    if nsys_metrics_path.exists():
        with open(nsys_metrics_path) as f:
            metrics = json.load(f)
    else:
        print(f"    {C['w']}No metrics file from nsys run{C['r']}")
        metrics = []

    # Try to parse nsys stats
    nsys_captured = {
        "nsys_version": nsys_version,
        "report_file": f"{report_name}.nsys-rep",
        "total_nsys_time_s": round(nsys_elapsed, 1),
    }

    # Check for generated report
    rep_file = Path(f"{report_name}.nsys-rep")
    if rep_file.exists():
        nsys_captured["report_size_mb"] = round(rep_file.stat().st_size / (1024*1024), 1)
        kv("Report size", f"{nsys_captured['report_size_mb']:.1f}", "MB")

    # Try to export and parse SQLite for kernel data
    sqlite_path = Path(f"{report_name}.sqlite")
    try:
        subprocess.run(
            [nsys_bin, "export", "--type", "sqlite", "--output", str(sqlite_path),
             str(rep_file)],
            capture_output=True, timeout=120,
        )
        if sqlite_path.exists():
            nsys_captured["sqlite_file"] = str(sqlite_path)
            nsys_captured.update(_parse_nsys_sqlite(sqlite_path))
    except Exception as e:
        print(f"    {C['w']}SQLite export failed: {e}{C['r']}")

    tps_values = [m["tok_s"] for m in metrics] if metrics else [0]

    return {
        "pass": "nsys",
        "total_time_s": round(nsys_elapsed, 3),
        "mean_tok_s": round(sum(tps_values) / len(tps_values), 2) if tps_values else 0,
        "min_tok_s": round(min(tps_values), 2) if tps_values else 0,
        "max_tok_s": round(max(tps_values), 2) if tps_values else 0,
        "total_tokens": sum(m["tokens"] for m in metrics) if metrics else 0,
        "per_request": metrics,
        "captured_data": nsys_captured,
    }


def _parse_nsys_sqlite(sqlite_path):
    """Extract kernel stats from nsys SQLite export."""
    import sqlite3

    result = {}
    try:
        conn = sqlite3.connect(str(sqlite_path))

        # Count kernels
        try:
            cur = conn.execute("""
                SELECT COUNT(*), COUNT(DISTINCT shortName)
                FROM CUPTI_ACTIVITY_KIND_KERNEL
            """)
            total, unique = cur.fetchone()
            result["total_kernel_launches"] = total
            result["unique_kernels"] = unique
        except Exception:
            pass

        # Total GPU time
        try:
            cur = conn.execute("""
                SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL
            """)
            total_ns = cur.fetchone()[0] or 0
            result["total_cuda_time_ms"] = round(total_ns / 1e6, 1)
        except Exception:
            pass

        # Memory ops
        try:
            cur = conn.execute("""
                SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """)
            result["memory_copy_ops"] = cur.fetchone()[0]
        except Exception:
            pass

        try:
            cur = conn.execute("""
                SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMSET
            """)
            result["memory_set_ops"] = cur.fetchone()[0]
        except Exception:
            pass

        conn.close()
    except Exception as e:
        result["parse_error"] = str(e)

    return result


# ================================================================
# KERNEL CATEGORIZATION
# ================================================================

_CAT_KEYWORDS = {
    "GEMM / MatMul": ["gemm", "cutlass", "cublas", "matmul", "sgemm", "hgemm",
                       "wmma", "mma_", "ampere_", "sm80_", "sm90_"],
    "Attention": ["attention", "flash_attn", "fmha", "sdpa", "flash_fwd"],
    "Normalization": ["layernorm", "rmsnorm", "layer_norm", "rms_norm"],
    "Activation": ["silu", "gelu", "relu", "swiglu", "sigmoid"],
    "Softmax": ["softmax"],
    "Memory": ["memcpy", "memset", "copy_kernel", "fill_kernel"],
    "Communication": ["nccl", "all_reduce"],
    "KV Cache": ["reshape_and_cache", "paged_attention"],
}

def _categorize(name):
    nl = name.lower()
    for cat, kws in _CAT_KEYWORDS.items():
        for kw in kws:
            if kw in nl:
                return cat
    return "Other"


# ================================================================
# COMPARISON REPORT
# ================================================================

def print_comparison(baseline, kineto, nsys, output_dir):
    """Print and generate the comparison report."""

    banner("COMPARISON RESULTS", "▓")

    # ── Throughput table ──
    section("Throughput & Overhead")

    passes = [("Baseline", baseline), ("Kineto", kineto)]
    if nsys:
        passes.append(("Nsys", nsys))

    print(f"\n    {C['d']}{'Tool':<14} {'Mean tok/s':>12} {'Min':>10} {'Max':>10} "
          f"{'Total Time':>12} {'Overhead':>10}{C['r']}")
    print(f"    {C['d']}{'─' * 72}{C['r']}")

    b_mean = baseline["mean_tok_s"]
    for label, data in passes:
        if data is None:
            continue
        overhead = ((b_mean - data["mean_tok_s"]) / b_mean * 100) if b_mean > 0 else 0
        oh_str = f"{overhead:+.1f}%" if label != "Baseline" else "—"
        color = C['w'] if overhead > 5 else C['ok'] if overhead < 2 else C['v']

        print(f"    {C['v']}{label:<14}{C['r']}"
              f" {data['mean_tok_s']:>12.1f}"
              f" {data['min_tok_s']:>10.1f}"
              f" {data['max_tok_s']:>10.1f}"
              f" {data['total_time_s']:>11.1f}s"
              f" {color}{oh_str:>10}{C['r']}")

    # ── Data Capture Comparison ──
    section("Captured Data Comparison")

    kineto_data = kineto.get("captured_data", {})
    nsys_data = nsys.get("captured_data", {}) if nsys else {}

    rows = [
        ("Unique kernels", kineto_data.get("unique_kernels", "—"),
         nsys_data.get("unique_kernels", "—")),
        ("Total kernel launches", kineto_data.get("total_kernel_launches", "—"),
         nsys_data.get("total_kernel_launches", "—")),
        ("Total CUDA time (ms)", kineto_data.get("total_cuda_time_ms", "—"),
         nsys_data.get("total_cuda_time_ms", "—")),
        ("Memory events", kineto_data.get("memory_events", "—"),
         f"{nsys_data.get('memory_copy_ops', 0)} + {nsys_data.get('memory_set_ops', 0)}"
         if nsys_data else "—"),
        ("Trace file size (MB)", kineto_data.get("trace_size_mb", "—"),
         nsys_data.get("report_size_mb", "—")),
    ]

    hdr = f"{'Metric':<28} {'Kineto':>16} {'Nsys':>16}"
    print(f"\n    {C['d']}{hdr}{C['r']}")
    print(f"    {C['d']}{'─' * 62}{C['r']}")

    for label, k_val, n_val in rows:
        print(f"    {C['v']}{label:<28}{C['r']} {str(k_val):>16} {str(n_val):>16}")

    # ── Kineto category breakdown ──
    if kineto_data.get("categories"):
        section("Kineto — GPU Time by Category")
        total_ms = sum(kineto_data["categories"].values())
        for cat, ms in kineto_data["categories"].items():
            pct = ms / total_ms * 100 if total_ms > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"    {C['v']}{cat:<22}{C['r']} {ms:>8.1f} ms  {pct:>5.1f}%  {C['b']}{bar}{C['r']}")

    # ── Capability Matrix ──
    section("Capability Matrix")

    capabilities = [
        ("CUDA kernel tracing",           "✓", "✓"),
        ("CPU operator tracing",          "✓", "✓ (sampling)"),
        ("Operator → kernel correlation", "✓", "✗ (needs NVTX)"),
        ("Memory alloc tracking",         "✓", "✓"),
        ("Memory bandwidth analysis",     "✗", "✓"),
        ("PCIe transfer tracking",        "✗", "✓"),
        ("NVLink traffic",                "✗", "✓"),
        ("GPU context switches",          "✗", "✓"),
        ("Python stack traces",           "✓", "✗"),
        ("Module hierarchy",              "✓", "✗"),
        ("FLOPS estimation",              "✓", "✗"),
        ("Input shape recording",         "✓", "✗"),
        ("Chrome trace export",           "✓", "✗ (nsys-ui)"),
        ("TensorBoard integration",       "✓", "✗"),
        ("Requires external tool",        "✗ (built-in)", "✓ (nsys binary)"),
        ("Requires root/sudo",            "✗", "Sometimes"),
        ("Embeddable in Python",          "✓", "✗ (wraps process)"),
        ("System-wide profiling",         "✗", "✓"),
        ("Multi-process support",         "Limited", "✓"),
        ("Kernel efficiency metrics",     "✗", "✗ (use ncu)"),
        ("Hardware counters",             "✗", "✗ (use ncu)"),
    ]

    print(f"\n    {C['d']}{'Capability':<35} {'Kineto':>18} {'Nsight Systems':>18}{C['r']}")
    print(f"    {C['d']}{'─' * 74}{C['r']}")

    for cap, k, n in capabilities:
        k_color = C['ok'] if k.startswith("✓") else C['e'] if k.startswith("✗") else C['w']
        n_color = C['ok'] if n.startswith("✓") else C['e'] if n.startswith("✗") else C['w']
        print(f"    {C['v']}{cap:<35}{C['r']} {k_color}{k:>18}{C['r']} {n_color}{n:>18}{C['r']}")

    # ── Recommendations ──
    section("Recommendations")

    b_tok = baseline["mean_tok_s"]
    k_tok = kineto["mean_tok_s"]
    k_overhead = ((b_tok - k_tok) / b_tok * 100) if b_tok > 0 else 0

    print(f"\n    Kineto overhead: {C['w']}{k_overhead:+.1f}%{C['r']} throughput impact")

    if nsys:
        n_tok = nsys["mean_tok_s"]
        n_overhead = ((b_tok - n_tok) / b_tok * 100) if b_tok > 0 else 0
        print(f"    Nsys overhead:   {C['w']}{n_overhead:+.1f}%{C['r']} throughput impact")

        if k_overhead < n_overhead:
            print(f"\n    {C['ok']}→ Kineto has lower overhead than nsys for this workload.{C['r']}")
        else:
            print(f"\n    {C['ok']}→ Nsys has lower overhead than Kineto for this workload.{C['r']}")

    print(f"""
    {C['d']}Use Kineto when:{C['r']}
      • You need operator-level insight (which PyTorch ops are slow)
      • You want Python stack traces correlating ops to model code
      • You need to embed profiling in existing Python benchmarks
      • You want Chrome/Perfetto trace visualization without nsys-ui

    {C['d']}Use Nsight Systems when:{C['r']}
      • You need system-wide GPU profiling (PCIe, NVLink, multi-process)
      • You need accurate GPU memory bandwidth measurements
      • You're profiling non-PyTorch CUDA code
      • You need the official NVIDIA profiling GUI (nsys-ui)

    {C['d']}Use Both when:{C['r']}
      • You want operator→kernel correlation with system context
      • Research paper needs comprehensive profiling methodology
    """)

    # ── Save JSON report ──
    section("Saving Report")

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": baseline.get("model", "unknown"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "comparison": {
            "baseline": baseline,
            "kineto": kineto,
            "nsys": nsys,
        },
        "overhead_summary": {
            "kineto_throughput_overhead_pct": round(k_overhead, 2),
            "nsys_throughput_overhead_pct": round(
                ((b_tok - nsys["mean_tok_s"]) / b_tok * 100), 2
            ) if nsys and b_tok > 0 else None,
        },
        "capability_matrix": {cap: {"kineto": k, "nsys": n}
                              for cap, k, n in capabilities},
    }

    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    kv("Report", str(report_path))

    return report


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kineto vs Nsight Systems — Overhead & Capability Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 kineto_vs_nsys_comparison.py --model Qwen/Qwen2.5-7B-Instruct --requests 5
  python3 kineto_vs_nsys_comparison.py --model Qwen/Qwen2.5-7B-Instruct --skip-nsys
        """,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max output tokens (keep low for faster comparison)")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./results/kineto_comparison")
    parser.add_argument("--skip-nsys", action="store_true",
                        help="Skip nsys pass (useful if nsys not available)")

    args = parser.parse_args()

    # ── Set VLLM_TORCH_PROFILER_DIR BEFORE LLM init ──
    model_safe = args.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{model_safe}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_dir = output_dir / "kineto_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str(trace_dir)

    from vllm import LLM, SamplingParams

    banner("KINETO vs NSIGHT SYSTEMS — OVERHEAD COMPARISON", "▓")
    print(f"\n    {C['d']}Model:           {args.model}{C['r']}")
    print(f"    {C['d']}Requests:        {args.requests} (warmup: {args.warmup}){C['r']}")
    print(f"    {C['d']}Max Tokens:      {args.max_tokens}{C['r']}")
    print(f"    {C['d']}Tensor Parallel: {args.tensor_parallel}{C['r']}")
    print(f"    {C['d']}Output:          {output_dir}{C['r']}")
    print(f"    {C['d']}PyTorch:         {torch.__version__}{C['r']}")
    print(f"    {C['d']}CUDA:            {torch.version.cuda}{C['r']}")
    print(f"    {C['d']}GPU:             {torch.cuda.device_count()}× {torch.cuda.get_device_name(0)}{C['r']}")
    print(f"    {C['d']}nsys pass:       {'skip' if args.skip_nsys else 'enabled'}{C['r']}")
    print(f"    {C['d']}Profiling via:   VLLM_TORCH_PROFILER_DIR + start_profile/stop_profile{C['r']}")

    # Load model (shared for baseline + kineto)
    section("Loading Model")
    load_start = time.monotonic()

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel,
        "trust_remote_code": True,
        "max_model_len": 2048,
        "enforce_eager": True,
    }
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    llm = LLM(**llm_kwargs)
    load_elapsed = time.monotonic() - load_start
    kv("Model loaded in", f"{load_elapsed:.1f}", "s")

    sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)

    all_prompts = get_prompts(args.requests)

    # Warmup
    if args.warmup > 0:
        section(f"Warmup ({args.warmup} requests)")
        for i in range(args.warmup):
            _ = llm.generate([all_prompts[i]], sampling_params)
        print(f"    {C['ok']}Warmup complete{C['r']}")

    measured_prompts = all_prompts[args.warmup:]

    # ── Pass 1: Baseline ──
    baseline_results = run_baseline(llm, measured_prompts, sampling_params)

    # ── Pass 2: Kineto ──
    kineto_results = run_kineto(llm, measured_prompts, sampling_params, output_dir)

    # ── Pass 3: Nsys ──
    nsys_results = None
    if not args.skip_nsys:
        nsys_results = run_nsys(llm, measured_prompts, sampling_params, output_dir, args)

    # ── Comparison ──
    report = print_comparison(baseline_results, kineto_results, nsys_results, output_dir)

    banner("COMPARISON COMPLETE", "▓")
    print(f"\n    {C['d']}All results saved to: {output_dir}{C['r']}")
    print()


if __name__ == "__main__":
    main()
