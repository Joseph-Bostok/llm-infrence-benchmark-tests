#!/usr/bin/env python3
"""
PInsight Unified Profiling Suite
==================================
Runs LLM inference through multiple backends and profilers,
producing clean, consistent, comparable output.

Profiling Levels:
  L1 — Application Metrics    (TTFT, ITL, TPOT, throughput)
  L2 — Framework Internals    (scheduling, batching, tokenization)
  L3 — CUDA Runtime           (kernel launches, memory ops, sync)
  L4 — GPU Kernels            (individual kernel timing, occupancy)

Backends:
  vllm   — Python/PyTorch inference (full L1-L4 visibility)
  ollama — Go/llama.cpp inference   (L1 + L3-L4 via nsys)

Usage:
    # Profile vLLM only
    python3 profile_suite.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct

    # Profile Ollama only
    python3 profile_suite.py --backend ollama --model qwen2.5:7b

    # Profile both for comparison
    python3 profile_suite.py --backend both --vllm-model Qwen/Qwen2.5-7B-Instruct --ollama-model qwen2.5:7b

    # With nsys GPU tracing (wraps the run)
    python3 profile_suite.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct --nsys

    # Quick test
    python3 profile_suite.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct --requests 3 --max-tokens 64
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ================================================================
# CONSTANTS
# ================================================================

PROMPTS = [
    "Explain the concept of recursion in programming in 3 sentences.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis in simple terms.",
    "Write a brief overview of the French Revolution.",
    "Explain how a neural network learns from data.",
]

COLORS = {
    "header": "\033[1;36m",   # Bold Cyan
    "section": "\033[1;33m",  # Bold Yellow
    "ok": "\033[0;32m",       # Green
    "warn": "\033[1;33m",     # Yellow
    "err": "\033[0;31m",      # Red
    "dim": "\033[0;90m",      # Gray
    "val": "\033[1;37m",      # Bold White
    "unit": "\033[0;37m",     # White
    "reset": "\033[0m",
}

C = COLORS

# ================================================================
# DISPLAY HELPERS
# ================================================================

def banner(text, char="═"):
    w = 70
    pad = (w - len(text) - 2) // 2
    print(f"\n{C['header']}{char * pad} {text} {char * pad}{C['reset']}")

def section(text):
    print(f"\n{C['section']}── {text} {'─' * (66 - len(text))}{C['reset']}")

def kv(key, value, unit="", indent=4):
    pad = " " * indent
    print(f"{pad}{C['dim']}{key:<28}{C['reset']}{C['val']}{value}{C['reset']} {C['unit']}{unit}{C['reset']}")

def table_header(cols, widths):
    header = ""
    for col, w in zip(cols, widths):
        header += f"{C['dim']}{col:>{w}}{C['reset']}  "
    print(f"    {header}")
    print(f"    {C['dim']}{'─' * (sum(widths) + 2 * len(widths))}{C['reset']}")

def table_row(vals, widths, highlight_idx=None):
    row = ""
    for i, (val, w) in enumerate(zip(vals, widths)):
        if i == highlight_idx:
            row += f"{C['ok']}{val:>{w}}{C['reset']}  "
        elif i == 0:
            row += f"{C['val']}{val:<{w}}{C['reset']}  "
        else:
            row += f"{val:>{w}}  "
    print(f"    {row}")

def pct_bar(pct, width=30):
    filled = int(pct / 100 * width)
    return f"{'█' * filled}{'░' * (width - filled)}"


# ================================================================
# VLLM BACKEND
# ================================================================

def profile_vllm(model, requests, warmup, max_tokens, tp, hf_token, output_dir):
    """Profile vLLM inference with detailed per-request metrics."""
    import torch
    from vllm import LLM, SamplingParams

    results = {
        "backend": "vllm",
        "model": model,
        "profiler": "python-timing",
        "config": {
            "tensor_parallel": tp,
            "max_tokens": max_tokens,
            "requests": requests,
            "warmup": warmup,
            "dtype": "bfloat16",
            "eager_mode": True,
        },
        "timing": {},
        "requests": [],
        "aggregate": {},
    }

    banner("vLLM BACKEND")

    section("Configuration")
    kv("Model", model)
    kv("Tensor Parallel", tp)
    kv("Requests", f"{requests}", f"({warmup} warmup + {requests - warmup} measured)")
    kv("Max Tokens", max_tokens)

    # --- Load Model ---
    section("Model Loading")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    load_start = time.monotonic()
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        max_model_len=2048,
        enforce_eager=True,
    )
    load_elapsed = time.monotonic() - load_start
    results["timing"]["model_load_s"] = round(load_elapsed, 2)

    kv("Load Time", f"{load_elapsed:.1f}", "seconds")

    try:
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated() / (1024**3)
        mem_reserved = torch.cuda.memory_reserved() / (1024**3)
        kv("GPU Memory (main proc)", f"{mem_alloc:.2f} / {mem_reserved:.2f}", "GB alloc / reserved")
        kv("Note", "Model weights are in EngineCore subprocess")
    except Exception:
        pass

    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(requests)]

    # --- Warmup ---
    section("Warmup")
    for i in range(warmup):
        t0 = time.monotonic()
        _ = llm.generate([prompts[i]], sampling_params)
        t1 = time.monotonic()
        kv(f"Warmup {i+1}", f"{t1 - t0:.2f}", "seconds")

    # --- Measured Inference ---
    section("Inference")
    measured = []
    for i in range(warmup, requests):
        prompt = prompts[i]
        t_start = time.monotonic()
        outputs = llm.generate([prompt], sampling_params)
        t_end = time.monotonic()

        output = outputs[0]
        num_tokens = len(output.outputs[0].token_ids)
        total_s = t_end - t_start
        tps = num_tokens / total_s if total_s > 0 else 0

        req_data = {
            "index": i,
            "prompt": prompt[:60] + "...",
            "tokens": num_tokens,
            "time_s": round(total_s, 4),
            "tokens_per_s": round(tps, 2),
        }
        measured.append(req_data)
        results["requests"].append(req_data)

        kv(f"Request {i}", f"{num_tokens} tok / {total_s:.2f}s", f"= {tps:.1f} tok/s")

    # --- Aggregates ---
    tps_vals = [r["tokens_per_s"] for r in measured]
    time_vals = [r["time_s"] for r in measured]
    tok_vals = [r["tokens"] for r in measured]

    results["aggregate"] = {
        "mean_tok_s": round(sum(tps_vals) / len(tps_vals), 2),
        "min_tok_s": round(min(tps_vals), 2),
        "max_tok_s": round(max(tps_vals), 2),
        "mean_time_s": round(sum(time_vals) / len(time_vals), 3),
        "total_tokens": sum(tok_vals),
    }

    section("Summary")
    kv("Mean Throughput", f"{results['aggregate']['mean_tok_s']}", "tok/s")
    kv("Range", f"{results['aggregate']['min_tok_s']} — {results['aggregate']['max_tok_s']}", "tok/s")
    kv("Mean Time/Request", f"{results['aggregate']['mean_time_s']}", "seconds")
    kv("Total Tokens", f"{results['aggregate']['total_tokens']}")

    return results


# ================================================================
# OLLAMA BACKEND
# ================================================================

def profile_ollama(model, requests, warmup, max_tokens, output_dir):
    """Profile Ollama inference with per-token timing."""
    import requests as req

    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    results = {
        "backend": "ollama",
        "model": model,
        "profiler": "python-timing",
        "config": {
            "ollama_host": OLLAMA_HOST,
            "max_tokens": max_tokens,
            "requests": requests,
            "warmup": warmup,
        },
        "timing": {},
        "requests": [],
        "aggregate": {},
    }

    banner("OLLAMA BACKEND")

    section("Configuration")
    kv("Model", model)
    kv("Ollama Host", OLLAMA_HOST)
    kv("Requests", f"{requests}", f"({warmup} warmup + {requests - warmup} measured)")
    kv("Max Tokens", max_tokens)

    # --- Check server ---
    section("Server Check")
    try:
        r = req.get(f"{OLLAMA_HOST}", timeout=5)
        kv("Status", "Running")
    except Exception:
        print(f"    {C['err']}ERROR: Ollama not reachable at {OLLAMA_HOST}{C['reset']}")
        return None

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(requests)]

    # --- Warmup (with cold start measurement) ---
    section("Warmup")
    for i in range(warmup):
        t0 = time.monotonic()

        # Stream to capture first token time
        first_token_time = None
        token_count = 0
        token_latencies = []

        resp = req.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": model,
            "prompt": prompts[i],
            "stream": True,
            "options": {"num_predict": max_tokens}
        }, stream=True)

        for line in resp.iter_lines():
            if line:
                t_tok = time.monotonic()
                data = json.loads(line)
                if data.get("response", ""):
                    if first_token_time is None:
                        first_token_time = t_tok
                    else:
                        token_latencies.append(t_tok)
                    token_count += 1
                if data.get("done", False):
                    break

        t1 = time.monotonic()
        ttft = (first_token_time - t0) * 1000 if first_token_time else 0
        kv(f"Warmup {i+1} (cold)", f"{t1 - t0:.2f}s", f"TTFT={ttft:.0f}ms, {token_count} tokens")

    # --- Measured Inference ---
    section("Inference")
    measured = []

    for i in range(warmup, requests):
        prompt = prompts[i]
        t_start = time.monotonic()

        first_token_time = None
        last_token_time = None
        token_count = 0
        token_times = []

        resp = req.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens}
        }, stream=True)

        for line in resp.iter_lines():
            if line:
                t_tok = time.monotonic()
                data = json.loads(line)
                if data.get("response", ""):
                    token_times.append(t_tok)
                    if first_token_time is None:
                        first_token_time = t_tok
                    last_token_time = t_tok
                    token_count += 1
                if data.get("done", False):
                    break

        t_end = time.monotonic()

        # Compute metrics
        ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else 0
        total_s = t_end - t_start

        itl_values = []
        for j in range(1, len(token_times)):
            itl_values.append((token_times[j] - token_times[j-1]) * 1000)

        mean_itl = sum(itl_values) / len(itl_values) if itl_values else 0
        tps = token_count / total_s if total_s > 0 else 0

        if last_token_time and first_token_time and token_count > 1:
            tpot = (last_token_time - first_token_time) / (token_count - 1) * 1000
        else:
            tpot = 0

        req_data = {
            "index": i,
            "prompt": prompt[:60] + "...",
            "tokens": token_count,
            "time_s": round(total_s, 4),
            "ttft_ms": round(ttft_ms, 2),
            "mean_itl_ms": round(mean_itl, 2),
            "tpot_ms": round(tpot, 2),
            "tokens_per_s": round(tps, 2),
        }
        measured.append(req_data)
        results["requests"].append(req_data)

        kv(f"Request {i}",
           f"{token_count} tok / {total_s:.2f}s",
           f"TTFT={ttft_ms:.1f}ms  ITL={mean_itl:.2f}ms  {tps:.1f} tok/s")

    # --- Aggregates ---
    tps_vals = [r["tokens_per_s"] for r in measured]
    ttft_vals = [r["ttft_ms"] for r in measured]
    itl_vals = [r["mean_itl_ms"] for r in measured]
    tpot_vals = [r["tpot_ms"] for r in measured]

    results["aggregate"] = {
        "mean_tok_s": round(sum(tps_vals) / len(tps_vals), 2),
        "min_tok_s": round(min(tps_vals), 2),
        "max_tok_s": round(max(tps_vals), 2),
        "mean_ttft_ms": round(sum(ttft_vals) / len(ttft_vals), 2),
        "mean_itl_ms": round(sum(itl_vals) / len(itl_vals), 2),
        "mean_tpot_ms": round(sum(tpot_vals) / len(tpot_vals), 2),
        "total_tokens": sum(r["tokens"] for r in measured),
    }

    section("Summary")
    kv("Mean Throughput", f"{results['aggregate']['mean_tok_s']}", "tok/s")
    kv("Mean TTFT", f"{results['aggregate']['mean_ttft_ms']}", "ms")
    kv("Mean ITL", f"{results['aggregate']['mean_itl_ms']}", "ms")
    kv("Mean TPOT", f"{results['aggregate']['mean_tpot_ms']}", "ms")
    kv("Total Tokens", f"{results['aggregate']['total_tokens']}")

    return results


# ================================================================
# NSYS GPU PROFILING (wraps either backend)
# ================================================================

def run_nsys(backend, model, requests, max_tokens, tp, hf_token, output_dir):
    """Run nsys profiling and parse the SQLite output."""

    report_name = output_dir / f"nsys_{backend}_{model.replace('/', '_').replace(':', '_')}"

    banner(f"NSIGHT SYSTEMS — {backend.upper()}")

    section("Configuration")
    kv("Backend", backend)
    kv("Model", model)
    kv("Output", str(report_name) + ".nsys-rep")

    # Build the inner command
    if backend == "vllm":
        inner_cmd = [
            "python3", "profile_suite.py",
            "--backend", "vllm",
            "--model", model,
            "--requests", str(requests),
            "--max-tokens", str(max_tokens),
            "--tensor-parallel", str(tp),
            "--output-dir", str(output_dir / "inner_metrics"),
        ]
        if hf_token:
            inner_cmd += ["--hf-token", hf_token]
    else:
        inner_cmd = [
            "python3", "profile_suite.py",
            "--backend", "ollama",
            "--model", model,
            "--requests", str(requests),
            "--max-tokens", str(max_tokens),
            "--output-dir", str(output_dir / "inner_metrics"),
        ]

    nsys_cmd = [
        "nsys", "profile",
        "--trace-fork-before-exec=true",
        "--output", str(report_name),
        "--trace", "cuda,nvtx,cudnn,cublas,osrt",
        "--cuda-memory-usage", "true",
        "--gpuctxsw", "true",
        "--force-overwrite", "true",
        "--stats", "true",
        "--"
    ] + inner_cmd

    section("Running nsys")
    kv("Command", " ".join(nsys_cmd[:8]) + " ...")
    print()

    proc = subprocess.run(nsys_cmd, capture_output=False)

    if proc.returncode != 0:
        print(f"    {C['err']}nsys exited with code {proc.returncode}{C['reset']}")
        return None

    # --- Parse SQLite for kernel data ---
    sqlite_path = f"{report_name}.sqlite"
    nsys_results = {"profiler": "nsys", "backend": backend, "model": model}

    if Path(sqlite_path).exists():
        nsys_results["kernels"] = parse_nsys_kernels(sqlite_path)
        nsys_results["memory"] = parse_nsys_memory(sqlite_path)

        # Print kernel summary
        section("GPU Kernel Summary")
        if nsys_results["kernels"]:
            total_gpu_ns = sum(k["total_ns"] for k in nsys_results["kernels"])
            cols = ["Kernel", "Calls", "Total(ms)", "Avg(μs)", "GPU%", "Distribution"]
            widths = [42, 7, 10, 10, 6, 30]
            table_header(cols, widths)

            for k in nsys_results["kernels"][:15]:
                pct = k["total_ns"] / total_gpu_ns * 100 if total_gpu_ns > 0 else 0
                table_row([
                    k["short_name"][:42],
                    str(k["calls"]),
                    f"{k['total_ns']/1e6:.1f}",
                    f"{k['avg_ns']/1e3:.1f}",
                    f"{pct:.1f}%",
                    pct_bar(pct),
                ], widths)

            print()
            kv("Total GPU Kernel Time", f"{total_gpu_ns/1e6:.1f}", "ms")
            kv("Unique Kernels", f"{len(nsys_results['kernels'])}")

        # Print memory summary
        section("GPU Memory Transfers")
        if nsys_results["memory"]:
            cols = ["Operation", "Count", "Total(MB)", "Total(ms)"]
            widths = [35, 8, 12, 12]
            table_header(cols, widths)

            for m in nsys_results["memory"]:
                table_row([
                    m["operation"],
                    str(m["count"]),
                    f"{m['total_bytes']/1e6:.1f}",
                    f"{m['total_ns']/1e6:.1f}",
                ], widths)

    return nsys_results


def parse_nsys_kernels(sqlite_path):
    """Extract kernel data from nsys SQLite export."""
    import sqlite3
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.execute("""
            SELECT shortName, COUNT(*) as calls,
                   SUM(end-start) as total_ns,
                   AVG(end-start) as avg_ns,
                   MIN(end-start) as min_ns,
                   MAX(end-start) as max_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY shortName
            ORDER BY SUM(end-start) DESC
            LIMIT 30
        """)
        kernels = []
        for row in cursor:
            kernels.append({
                "short_name": row[0],
                "calls": row[1],
                "total_ns": row[2],
                "avg_ns": row[3],
                "min_ns": row[4],
                "max_ns": row[5],
            })
        conn.close()
        return kernels
    except Exception as e:
        print(f"    {C['warn']}Could not parse kernels: {e}{C['reset']}")
        return []


def parse_nsys_memory(sqlite_path):
    """Extract memory transfer data from nsys SQLite export."""
    import sqlite3
    COPY_KINDS = {1: "Host → Device", 2: "Device → Host", 8: "Device → Device"}
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.execute("""
            SELECT copyKind, COUNT(*) as count,
                   SUM(bytes) as total_bytes,
                   SUM(end-start) as total_ns
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            GROUP BY copyKind
        """)
        memory = []
        for row in cursor:
            memory.append({
                "operation": COPY_KINDS.get(row[0], f"Kind {row[0]}"),
                "count": row[1],
                "total_bytes": row[2],
                "total_ns": row[3],
            })
        conn.close()
        return sorted(memory, key=lambda m: m["total_ns"], reverse=True)
    except Exception as e:
        print(f"    {C['warn']}Could not parse memory: {e}{C['reset']}")
        return []


# ================================================================
# COMPARISON
# ================================================================

def print_comparison(vllm_results, ollama_results):
    """Side-by-side comparison of vLLM vs Ollama."""
    banner("BACKEND COMPARISON: vLLM vs Ollama")

    section("Throughput")
    cols = ["Metric", "vLLM", "Ollama", "Δ"]
    widths = [24, 14, 14, 14]
    table_header(cols, widths)

    va = vllm_results.get("aggregate", {})
    oa = ollama_results.get("aggregate", {})

    comparisons = [
        ("Mean Throughput", "mean_tok_s", "tok/s", True),
        ("Total Tokens", "total_tokens", "", True),
    ]

    for label, key, unit, higher_better in comparisons:
        vv = va.get(key, 0)
        ov = oa.get(key, 0)
        if vv and ov:
            delta = ((vv - ov) / ov) * 100
            delta_str = f"{'+' if delta > 0 else ''}{delta:.1f}%"
        else:
            delta_str = "—"

        table_row([
            f"{label} ({unit})" if unit else label,
            str(vv), str(ov), delta_str,
        ], widths)

    # Ollama-specific latency metrics
    if "mean_ttft_ms" in oa:
        section("Latency (Ollama only — vLLM offline mode lacks per-token streaming)")
        kv("Mean TTFT", f"{oa['mean_ttft_ms']}", "ms")
        kv("Mean ITL", f"{oa['mean_itl_ms']}", "ms")
        kv("Mean TPOT", f"{oa['mean_tpot_ms']}", "ms")


def print_nsys_comparison(vllm_nsys, ollama_nsys):
    """Compare GPU kernel profiles between backends."""
    if not vllm_nsys or not ollama_nsys:
        return

    banner("GPU KERNEL COMPARISON: vLLM vs Ollama")

    vk = {k["short_name"]: k for k in vllm_nsys.get("kernels", [])}
    ok = {k["short_name"]: k for k in ollama_nsys.get("kernels", [])}

    all_kernels = set(list(vk.keys())[:10] + list(ok.keys())[:10])

    section("Top Kernels — Total GPU Time (ms)")
    cols = ["Kernel", "vLLM", "Ollama", "Shared?"]
    widths = [42, 12, 12, 8]
    table_header(cols, widths)

    # Sort by vLLM time
    sorted_kernels = sorted(all_kernels,
        key=lambda n: vk.get(n, {}).get("total_ns", 0), reverse=True)

    for name in sorted_kernels[:15]:
        v_time = vk.get(name, {}).get("total_ns", 0) / 1e6
        o_time = ok.get(name, {}).get("total_ns", 0) / 1e6
        shared = "✓" if name in vk and name in ok else "—"

        table_row([
            name[:42],
            f"{v_time:.1f}" if v_time > 0 else "—",
            f"{o_time:.1f}" if o_time > 0 else "—",
            shared,
        ], widths)

    # Memory comparison
    section("Memory Transfers Comparison")
    vm = {m["operation"]: m for m in vllm_nsys.get("memory", [])}
    om = {m["operation"]: m for m in ollama_nsys.get("memory", [])}

    cols = ["Transfer Type", "vLLM (MB)", "Ollama (MB)", "vLLM (ms)", "Ollama (ms)"]
    widths = [22, 12, 12, 12, 12]
    table_header(cols, widths)

    for op in ["Host → Device", "Device → Host", "Device → Device"]:
        v_mb = vm.get(op, {}).get("total_bytes", 0) / 1e6
        o_mb = om.get(op, {}).get("total_bytes", 0) / 1e6
        v_ms = vm.get(op, {}).get("total_ns", 0) / 1e6
        o_ms = om.get(op, {}).get("total_ns", 0) / 1e6

        table_row([
            op,
            f"{v_mb:.1f}" if v_mb > 0 else "—",
            f"{o_mb:.1f}" if o_mb > 0 else "—",
            f"{v_ms:.1f}" if v_ms > 0 else "—",
            f"{o_ms:.1f}" if o_ms > 0 else "—",
        ], widths)


# ================================================================
# SAVE RESULTS
# ================================================================

def save_results(all_results, output_dir):
    """Save all results to structured JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for key, data in all_results.items():
        if data is None:
            continue
        filepath = output_dir / f"{key}_{timestamp}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        kv(f"Saved {key}", str(filepath), indent=2)


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PInsight Unified Profiling Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=["vllm", "ollama", "both"],
                        default="vllm", help="Backend to profile")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (HF for vllm, tag for ollama)")
    parser.add_argument("--vllm-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model for vLLM")
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:7b",
                        help="Ollama model tag")
    parser.add_argument("--requests", type=int, default=5,
                        help="Total requests per backend")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup requests")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max output tokens")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Tensor parallel degree (vLLM only)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for gated models")
    parser.add_argument("--nsys", action="store_true",
                        help="Also run nsys GPU profiling")
    parser.add_argument("--output-dir", type=str, default="./results/profiling/unified",
                        help="Output directory")

    args = parser.parse_args()
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model names
    if args.model:
        if args.backend == "vllm":
            args.vllm_model = args.model
        elif args.backend == "ollama":
            args.ollama_model = args.model

    all_results = {}

    banner("PInsight UNIFIED PROFILING SUITE", "▓")
    print(f"\n    {C['dim']}CCI Aries Server · 4× A100-PCIE-40GB · Ollama + vLLM{C['reset']}")
    print(f"    {C['dim']}Timestamp: {datetime.now().isoformat()}{C['reset']}")
    print(f"    {C['dim']}Output:    {output_dir}{C['reset']}")

    # --- vLLM ---
    if args.backend in ("vllm", "both"):
        vllm_results = profile_vllm(
            args.vllm_model, args.requests, args.warmup,
            args.max_tokens, args.tensor_parallel, args.hf_token, output_dir,
        )
        all_results["vllm_metrics"] = vllm_results

        if args.nsys:
            nsys_results = run_nsys(
                "vllm", args.vllm_model, args.requests, args.max_tokens,
                args.tensor_parallel, args.hf_token, output_dir,
            )
            all_results["vllm_nsys"] = nsys_results

    # --- Ollama ---
    if args.backend in ("ollama", "both"):
        ollama_results = profile_ollama(
            args.ollama_model, args.requests, args.warmup,
            args.max_tokens, output_dir,
        )
        all_results["ollama_metrics"] = ollama_results

        if args.nsys:
            nsys_results = run_nsys(
                "ollama", args.ollama_model, args.requests, args.max_tokens,
                1, None, output_dir,
            )
            all_results["ollama_nsys"] = nsys_results

    # --- Comparison ---
    if args.backend == "both":
        vr = all_results.get("vllm_metrics")
        olr = all_results.get("ollama_metrics")
        if vr and olr:
            print_comparison(vr, olr)

        vn = all_results.get("vllm_nsys")
        on = all_results.get("ollama_nsys")
        if vn and on:
            print_nsys_comparison(vn, on)

    # --- Save ---
    banner("SAVING RESULTS")
    save_results(all_results, output_dir)

    banner("COMPLETE", "▓")
    print(f"\n    All results saved to: {output_dir}")
    print(f"    {C['dim']}To re-run with nsys: python3 profile_suite.py --backend {args.backend} --nsys{C['reset']}")
    print()


if __name__ == "__main__":
    main()
