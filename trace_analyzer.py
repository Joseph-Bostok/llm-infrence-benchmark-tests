#!/usr/bin/env python3
"""
PInsight Trace Analyzer
========================
Parses existing nsys SQLite exports and produces clean, formatted
profiling reports. Works with data already collected — no re-profiling needed.

Usage:
    # Analyze a single nsys trace
    python3 trace_analyzer.py results/profiling/nsys_20260217_171216/nsys_vllm_Qwen_Qwen2.5-7B-Instruct.sqlite

    # Compare two traces (vLLM vs Ollama)
    python3 trace_analyzer.py \
        --vllm results/profiling/nsys_*/nsys_vllm_*.sqlite \
        --ollama results/profiling/nsys_*/nsys_ollama_*.sqlite

    # Export to JSON for dashboard
    python3 trace_analyzer.py *.sqlite --export results.json
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# ================================================================
# COLORS
# ================================================================

C = {
    "h": "\033[1;36m", "s": "\033[1;33m", "ok": "\033[0;32m",
    "w": "\033[1;33m", "e": "\033[0;31m", "d": "\033[0;90m",
    "v": "\033[1;37m", "u": "\033[0;37m", "r": "\033[0m",
    "bar_fill": "\033[0;36m", "bar_empty": "\033[0;90m",
}

def banner(text, char="═"):
    w = 74
    pad = (w - len(text) - 2) // 2
    print(f"\n{C['h']}{char * pad} {text} {char * pad}{C['r']}")

def section(text):
    print(f"\n{C['s']}── {text} {'─' * (70 - len(text))}{C['r']}")

def kv(key, value, unit="", indent=4):
    print(f"{' ' * indent}{C['d']}{key:<30}{C['r']}{C['v']}{value}{C['r']} {C['u']}{unit}{C['r']}")

def pct_bar(pct, width=25):
    filled = int(pct / 100 * width)
    return f"{C['bar_fill']}{'█' * filled}{C['bar_empty']}{'░' * (width - filled)}{C['r']}"


# ================================================================
# KERNEL CATEGORIES
# ================================================================

def categorize_kernel(name):
    """Classify a CUDA kernel into an inference pipeline stage."""
    n = name.lower()
    if "gemm" in n or "gemv" in n or "cublas" in n or "matmul" in n:
        if "reduce" in n or "splitk" in n:
            return "GEMM Reduction", "Post-GEMM accumulation"
        return "GEMM / MatMul", "Matrix multiply (weights × activations)"
    elif "flash" in n or "attention" in n or "attn" in n or "fmha" in n:
        return "Attention", "Self-attention / Flash Attention"
    elif "rms_norm" in n or "layer_norm" in n or "norm" in n:
        return "Normalization", "RMSNorm / LayerNorm"
    elif "silu" in n or "act_and_mul" in n or "gelu" in n or "relu" in n:
        return "Activation", "SiLU / GeLU / ReLU (FFN)"
    elif "rotary" in n or "rope" in n:
        return "Positional Enc.", "Rotary Position Embedding"
    elif "reshape_and_cache" in n or "kv_cache" in n or "cache" in n:
        return "KV Cache", "KV cache write / reshape"
    elif "softmax" in n:
        return "Softmax", "Attention softmax"
    elif "copy" in n or "memcpy" in n or "scatter" in n or "gather" in n:
        return "Data Movement", "Tensor copy / scatter / gather"
    elif "fill" in n or "memset" in n:
        return "Initialization", "Tensor fill / zero"
    elif "elementwise" in n or "vectorized" in n:
        return "Elementwise", "Pointwise / elementwise ops"
    elif "reduce" in n or "scan" in n:
        return "Reduction", "Sum / argmax / scan"
    elif "index" in n or "embedding" in n:
        return "Indexing", "Token embedding / indexing"
    elif "radix" in n or "sort" in n:
        return "Sorting", "Radix sort (sampling)"
    else:
        return "Other", name[:50]


# ================================================================
# SQLITE PARSING
# ================================================================

def parse_kernels(sqlite_path):
    """Extract kernel data from nsys SQLite."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.execute("""
        SELECT s.value, COUNT(*) as calls,
               SUM(k.end-k.start) as total_ns,
               AVG(k.end-k.start) as avg_ns,
               MIN(k.end-k.start) as min_ns,
               MAX(k.end-k.start) as max_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName = s.id
        GROUP BY s.value
        ORDER BY SUM(k.end-k.start) DESC
    """)
    kernels = []
    for row in cursor:
        cat, desc = categorize_kernel(row[0])
        kernels.append({
            "name": row[0],
            "category": cat,
            "description": desc,
            "calls": row[1],
            "total_ns": row[2],
            "avg_ns": row[3],
            "min_ns": row[4],
            "max_ns": row[5],
        })
    conn.close()
    return kernels


def parse_memory(sqlite_path):
    """Extract memory transfer data."""
    COPY_KINDS = {
        1: "Host → Device",
        2: "Device → Host",
        8: "Device → Device",
        10: "Peer → Peer",
    }
    conn = sqlite3.connect(sqlite_path)
    try:
        cursor = conn.execute("""
            SELECT copyKind, COUNT(*) as count,
                   SUM(m.bytes) as total_bytes,
                   AVG(m.bytes) as avg_bytes,
                   SUM(k.end-k.start) as total_ns
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            GROUP BY copyKind
            ORDER BY SUM(k.end-k.start) DESC
        """)
        memory = []
        for row in cursor:
            memory.append({
                "kind": row[0],
                "operation": COPY_KINDS.get(row[0], f"Unknown ({row[0]})"),
                "count": row[1],
                "total_bytes": row[2],
                "avg_bytes": row[3],
                "total_ns": row[4],
            })
    except Exception:
        memory = []
    conn.close()
    return memory


def parse_memset(sqlite_path):
    """Extract memset operations."""
    conn = sqlite3.connect(sqlite_path)
    try:
        cursor = conn.execute("""
            SELECT COUNT(*) as count,
                   SUM(m.bytes) as total_bytes,
                   SUM(k.end-k.start) as total_ns
            FROM CUPTI_ACTIVITY_KIND_MEMSET
        """)
        row = cursor.fetchone()
        if row and row[0] > 0:
            return {"count": row[0], "total_bytes": row[1], "total_ns": row[2]}
    except Exception:
        pass
    conn.close()
    return None


def parse_cuda_api(sqlite_path):
    """Extract CUDA API call summary."""
    conn = sqlite3.connect(sqlite_path)
    try:
        cursor = conn.execute("""
            SELECT 
                CASE nameId 
                    WHEN nameId THEN nameId 
                END,
                COUNT(*) as calls,
                SUM(k.end-k.start) as total_ns,
                AVG(k.end-k.start) as avg_ns
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            GROUP BY nameId
            ORDER BY SUM(k.end-k.start) DESC
            LIMIT 15
        """)
        api_calls = []
        for row in cursor:
            api_calls.append({
                "name_id": row[0],
                "calls": row[1],
                "total_ns": row[2],
                "avg_ns": row[3],
            })
        return api_calls
    except Exception:
        return []
    finally:
        conn.close()


# ================================================================
# REPORT GENERATION
# ================================================================

def print_report(sqlite_path, label=None):
    """Generate full profiling report from an nsys SQLite export."""
    path = Path(sqlite_path)
    if not path.exists():
        print(f"{C['e']}File not found: {sqlite_path}{C['r']}")
        return None

    name = label or path.stem
    banner(f"GPU PROFILE: {name}")

    kv("Source", str(path))
    kv("Size", f"{path.stat().st_size / 1e6:.1f}", "MB")

    # --- Parse all data ---
    kernels = parse_kernels(sqlite_path)
    memory = parse_memory(sqlite_path)
    memset = parse_memset(sqlite_path)

    total_gpu_ns = sum(k["total_ns"] for k in kernels)
    total_calls = sum(k["calls"] for k in kernels)

    # ==========================
    # SECTION 1: Overview
    # ==========================
    section("Overview")
    kv("Total GPU Kernel Time", f"{total_gpu_ns/1e6:.1f}", "ms")
    kv("Total Kernel Launches", f"{total_calls:,}")
    kv("Unique Kernel Types", f"{len(kernels)}")

    # ==========================
    # SECTION 2: By Category
    # ==========================
    section("Inference Pipeline Breakdown")

    categories = {}
    for k in kernels:
        cat = k["category"]
        if cat not in categories:
            categories[cat] = {"total_ns": 0, "calls": 0, "kernels": []}
        categories[cat]["total_ns"] += k["total_ns"]
        categories[cat]["calls"] += k["calls"]
        categories[cat]["kernels"].append(k["name"])

    sorted_cats = sorted(categories.items(), key=lambda x: x[1]["total_ns"], reverse=True)

    print(f"\n    {C['d']}{'Category':<20} {'Time(ms)':>10} {'Calls':>8} {'GPU%':>7}  Distribution{C['r']}")
    print(f"    {C['d']}{'─' * 80}{C['r']}")

    for cat_name, cat_data in sorted_cats:
        pct = cat_data["total_ns"] / total_gpu_ns * 100 if total_gpu_ns > 0 else 0
        if pct < 0.05:
            continue
        print(f"    {C['v']}{cat_name:<20}{C['r']}"
              f" {cat_data['total_ns']/1e6:>10.1f}"
              f" {cat_data['calls']:>8,}"
              f" {pct:>6.1f}%"
              f"  {pct_bar(pct)}")

    # ==========================
    # SECTION 3: Top Kernels
    # ==========================
    section("Top 15 CUDA Kernels by GPU Time")

    print(f"\n    {C['d']}{'#':>3} {'Kernel':<45} {'Calls':>7} {'Total(ms)':>10} {'Avg(μs)':>10} {'GPU%':>6}{C['r']}")
    print(f"    {C['d']}{'─' * 85}{C['r']}")

    for i, k in enumerate(kernels[:15]):
        pct = k["total_ns"] / total_gpu_ns * 100 if total_gpu_ns > 0 else 0
        short = k["name"][:45]
        print(f"    {C['d']}{i+1:>3}{C['r']}"
              f" {C['v']}{short:<45}{C['r']}"
              f" {k['calls']:>7,}"
              f" {k['total_ns']/1e6:>10.1f}"
              f" {k['avg_ns']/1e3:>10.1f}"
              f" {pct:>5.1f}%")

    # ==========================
    # SECTION 4: Prefill vs Decode Analysis
    # ==========================
    section("Prefill vs Decode Analysis")

    # Heuristic: large GEMM (>1ms avg) = prefill, small GEMM (<500μs avg) = decode
    prefill_ns = 0
    decode_ns = 0
    other_ns = 0

    for k in kernels:
        if k["category"] == "GEMM / MatMul":
            if k["avg_ns"] > 1_000_000:  # >1ms → prefill
                prefill_ns += k["total_ns"]
            else:
                decode_ns += k["total_ns"]
        elif k["category"] in ("Attention", "Normalization", "Activation",
                                "Positional Enc.", "KV Cache", "Softmax",
                                "GEMM Reduction"):
            decode_ns += k["total_ns"]
        else:
            other_ns += k["total_ns"]

    total_infer_ns = prefill_ns + decode_ns + other_ns
    if total_infer_ns > 0:
        kv("Prefill (large GEMM)", f"{prefill_ns/1e6:.1f}",
           f"ms  ({prefill_ns/total_infer_ns*100:.1f}%)")
        kv("Decode (small GEMM + pipeline)", f"{decode_ns/1e6:.1f}",
           f"ms  ({decode_ns/total_infer_ns*100:.1f}%)")
        kv("Other (init, sampling, etc.)", f"{other_ns/1e6:.1f}",
           f"ms  ({other_ns/total_infer_ns*100:.1f}%)")

    # ==========================
    # SECTION 5: Memory
    # ==========================
    section("GPU Memory Transfers")

    if memory:
        total_mem_bytes = sum(m["total_bytes"] for m in memory)
        total_mem_ns = sum(m["total_ns"] for m in memory)

        print(f"\n    {C['d']}{'Direction':<25} {'Count':>8} {'Total':>12} {'Time(ms)':>10} {'Bandwidth':>12}{C['r']}")
        print(f"    {C['d']}{'─' * 70}{C['r']}")

        for m in memory:
            bw = m["total_bytes"] / (m["total_ns"] / 1e9) / 1e9 if m["total_ns"] > 0 else 0
            size_str = f"{m['total_bytes']/1e6:.1f} MB" if m["total_bytes"] < 1e9 else f"{m['total_bytes']/1e9:.2f} GB"
            print(f"    {C['v']}{m['operation']:<25}{C['r']}"
                  f" {m['count']:>8,}"
                  f" {size_str:>12}"
                  f" {m['total_ns']/1e6:>10.1f}"
                  f" {bw:>10.1f} GB/s")

        print()
        kv("Total Transferred", f"{total_mem_bytes/1e9:.2f}", "GB")
        kv("Total Transfer Time", f"{total_mem_ns/1e6:.1f}", "ms")

    if memset:
        print()
        kv("Memset Operations", f"{memset['count']:,}",
           f"({memset['total_bytes']/1e6:.1f} MB in {memset['total_ns']/1e6:.1f} ms)")

    # Return structured data
    return {
        "label": name,
        "source": str(path),
        "total_gpu_time_ms": round(total_gpu_ns / 1e6, 1),
        "total_kernel_launches": total_calls,
        "unique_kernels": len(kernels),
        "categories": {cat: {
            "time_ms": round(data["total_ns"] / 1e6, 1),
            "calls": data["calls"],
            "pct": round(data["total_ns"] / total_gpu_ns * 100, 1) if total_gpu_ns > 0 else 0,
        } for cat, data in sorted_cats},
        "prefill_decode": {
            "prefill_ms": round(prefill_ns / 1e6, 1),
            "decode_ms": round(decode_ns / 1e6, 1),
            "other_ms": round(other_ns / 1e6, 1),
        },
        "top_kernels": [{
            "name": k["name"],
            "category": k["category"],
            "calls": k["calls"],
            "total_ms": round(k["total_ns"] / 1e6, 1),
            "avg_us": round(k["avg_ns"] / 1e3, 1),
            "pct": round(k["total_ns"] / total_gpu_ns * 100, 1) if total_gpu_ns > 0 else 0,
        } for k in kernels[:30]],
        "memory": [{
            "direction": m["operation"],
            "count": m["count"],
            "total_mb": round(m["total_bytes"] / 1e6, 1),
            "time_ms": round(m["total_ns"] / 1e6, 1),
        } for m in memory],
    }


def print_dual_comparison(data_a, data_b):
    """Side-by-side comparison of two traces."""
    banner(f"COMPARISON: {data_a['label']} vs {data_b['label']}")

    section("High-Level")
    print(f"\n    {C['d']}{'Metric':<35} {data_a['label']:>16} {data_b['label']:>16} {'Ratio':>10}{C['r']}")
    print(f"    {C['d']}{'─' * 80}{C['r']}")

    metrics = [
        ("Total GPU Time (ms)", data_a["total_gpu_time_ms"], data_b["total_gpu_time_ms"]),
        ("Kernel Launches", data_a["total_kernel_launches"], data_b["total_kernel_launches"]),
        ("Unique Kernels", data_a["unique_kernels"], data_b["unique_kernels"]),
        ("Prefill Time (ms)", data_a["prefill_decode"]["prefill_ms"], data_b["prefill_decode"]["prefill_ms"]),
        ("Decode Time (ms)", data_a["prefill_decode"]["decode_ms"], data_b["prefill_decode"]["decode_ms"]),
    ]

    for label, va, vb in metrics:
        ratio = f"{va/vb:.2f}x" if vb > 0 else "—"
        print(f"    {C['v']}{label:<35}{C['r']} {va:>16,} {vb:>16,} {ratio:>10}")

    section("Pipeline Category Comparison")
    all_cats = set(list(data_a["categories"].keys()) + list(data_b["categories"].keys()))
    sorted_cats = sorted(all_cats,
        key=lambda c: data_a["categories"].get(c, {}).get("time_ms", 0), reverse=True)

    print(f"\n    {C['d']}{'Category':<20} {data_a['label']+' (ms)':>14} {data_b['label']+' (ms)':>14} {'Ratio':>10}{C['r']}")
    print(f"    {C['d']}{'─' * 62}{C['r']}")

    for cat in sorted_cats:
        va = data_a["categories"].get(cat, {}).get("time_ms", 0)
        vb = data_b["categories"].get(cat, {}).get("time_ms", 0)
        ratio = f"{va/vb:.2f}x" if vb > 0 else "—" if va > 0 else "—"
        if va < 0.1 and vb < 0.1:
            continue
        print(f"    {C['v']}{cat:<20}{C['r']} {va:>14.1f} {vb:>14.1f} {ratio:>10}")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PInsight Trace Analyzer — parse nsys SQLite exports",
    )
    parser.add_argument("sqlite_files", nargs="*",
                        help="SQLite file(s) to analyze")
    parser.add_argument("--vllm", type=str, default=None,
                        help="vLLM nsys SQLite file")
    parser.add_argument("--ollama", type=str, default=None,
                        help="Ollama nsys SQLite file")
    parser.add_argument("--export", type=str, default=None,
                        help="Export results to JSON file")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels for each SQLite file")

    args = parser.parse_args()

    all_results = []

    # Mode 1: Named comparison (--vllm and --ollama)
    if args.vllm or args.ollama:
        if args.vllm:
            vllm_data = print_report(args.vllm, label="vLLM")
            if vllm_data:
                all_results.append(vllm_data)
        if args.ollama:
            ollama_data = print_report(args.ollama, label="Ollama")
            if ollama_data:
                all_results.append(ollama_data)
        if len(all_results) == 2:
            print_dual_comparison(all_results[0], all_results[1])

    # Mode 2: Positional files
    elif args.sqlite_files:
        for i, fpath in enumerate(args.sqlite_files):
            label = args.labels[i] if args.labels and i < len(args.labels) else None
            data = print_report(fpath, label=label)
            if data:
                all_results.append(data)

        if len(all_results) == 2:
            print_dual_comparison(all_results[0], all_results[1])

    else:
        parser.print_help()
        return

    # Export
    if args.export and all_results:
        with open(args.export, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n    {C['ok']}Exported to {args.export}{C['r']}")

    print()


if __name__ == "__main__":
    main()
