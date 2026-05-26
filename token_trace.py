#!/usr/bin/env python3
"""
PInsight Token-Level GPU Trace
================================
Extracts the temporal kernel trace from an nsys SQLite export and
reconstructs what the GPU does for EACH TOKEN during inference.

This shows the actual per-token pipeline:
  Token N: GEMM → GEMM → RMSNorm → Attention → RoPE → GEMM → SiLU → GEMM → ...
           Layer 0                                      Layer 1

Usage:
    # Show the per-token pipeline for the first 5 tokens
    python3 token_trace.py nsys_vllm_*.sqlite --tokens 5

    # Show all decode tokens
    python3 token_trace.py nsys_vllm_*.sqlite --tokens all

    # Export the full timeline to JSON
    python3 token_trace.py nsys_vllm_*.sqlite --export token_timeline.json

    # Show a single token in detail (every kernel)
    python3 token_trace.py nsys_vllm_*.sqlite --detail-token 10
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
    # Pipeline stage colors
    "gemm": "\033[1;31m",      # Red — matrix multiply
    "attn": "\033[1;34m",      # Blue — attention
    "norm": "\033[1;33m",      # Yellow — normalization
    "act": "\033[1;32m",       # Green — activation
    "rope": "\033[1;35m",      # Magenta — positional encoding
    "kv": "\033[1;36m",        # Cyan — KV cache
    "other": "\033[0;37m",     # White — everything else
}

STAGE_COLORS = {
    "GEMM": C["gemm"],
    "Attn": C["attn"],
    "Norm": C["norm"],
    "Act": C["act"],
    "RoPE": C["rope"],
    "KV": C["kv"],
    "Soft": C["other"],
    "Red": C["other"],
    "Other": C["d"],
}

def banner(text, char="═"):
    w = 74
    pad = (w - len(text) - 2) // 2
    print(f"\n{C['h']}{char * pad} {text} {char * pad}{C['r']}")

def section(text):
    print(f"\n{C['s']}── {text} {'─' * (70 - len(text))}{C['r']}")


# ================================================================
# KERNEL CLASSIFICATION
# ================================================================

def classify_kernel(name):
    """Classify kernel into pipeline stage with short label."""
    n = name.lower()
    if "gemm" in n or "gemv" in n or "matmul" in n:
        # Determine GEMM type from tile size
        if "128x256" in n or "256x128" in n or "128x128" in n:
            return "GEMM", "GEMM-L"   # Large = prefill
        elif "128x64" in n:
            return "GEMM", "GEMM-M"   # Medium = main decode
        elif "64x64" in n:
            return "GEMM", "GEMM-S"   # Small = decode projections
        return "GEMM", "GEMM"
    elif "flash" in n or "fmha" in n:
        if "combine" in n or "reduce" in n:
            return "Attn", "Attn-R"   # Attention reduction
        return "Attn", "Attn"
    elif "rms_norm" in n or "layer_norm" in n:
        if "fused_add" in n:
            return "Norm", "FNorm"    # Fused add + normalize
        return "Norm", "Norm"
    elif "silu" in n or "act_and_mul" in n or "gelu" in n:
        return "Act", "SiLU"
    elif "rotary" in n or "rope" in n:
        return "RoPE", "RoPE"
    elif "reshape_and_cache" in n or "kv_cache" in n:
        return "KV", "KV$"           # KV cache write
    elif "softmax" in n:
        return "Soft", "Smax"
    elif "splitkreduce" in n:
        return "Red", "SplK"         # Split-K reduction (post-GEMM)
    elif "radix" in n or "sort" in n:
        return "Other", "Sort"
    elif "elementwise" in n or "vectorized" in n:
        return "Other", "Elem"
    elif "reduce" in n or "argmax" in n:
        return "Other", "Rdce"
    elif "copy" in n or "scatter" in n:
        return "Other", "Copy"
    elif "index" in n or "embedding" in n:
        return "Other", "Idx"
    elif "fill" in n:
        return "Other", "Fill"
    else:
        return "Other", "?"


# ================================================================
# DATA EXTRACTION
# ================================================================

def extract_kernel_timeline(sqlite_path):
    """Extract all kernel events with timestamps, ordered by start time."""
    conn = sqlite3.connect(sqlite_path)

    # Get all kernels with their string names, ordered by start time
    cursor = conn.execute("""
        SELECT s.value as name, k.start, k.end,
               (k.end - k.start) as duration_ns,
               k.gridX, k.gridY, k.gridZ,
               k.blockX, k.blockY, k.blockZ
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start ASC
    """)

    events = []
    for row in cursor:
        stage, label = classify_kernel(row[0])
        events.append({
            "name": row[0],
            "start_ns": row[1],
            "end_ns": row[2],
            "duration_ns": row[3],
            "stage": stage,
            "label": label,
            "grid": (row[4], row[5], row[6]),
            "block": (row[7], row[8], row[9]),
        })

    conn.close()
    return events


def identify_phases(events):
    """Split timeline into prefill and decode phases based on kernel patterns."""
    if not events:
        return [], [], 0

    # Heuristic: prefill uses large GEMMs (>1ms), decode uses small ones (<500μs)
    # Find the transition point: last large GEMM before the steady-state small GEMMs begin
    
    # First pass: find where large GEMMs stop appearing regularly
    large_gemm_indices = []
    for i, e in enumerate(events):
        if e["stage"] == "GEMM" and e["duration_ns"] > 1_000_000:  # >1ms
            large_gemm_indices.append(i)

    if not large_gemm_indices:
        # No large GEMMs = no prefill detected (unlikely)
        return [], events, 0

    # The prefill/decode boundary is after the last cluster of large GEMMs
    # Look for a gap where large GEMMs stop
    last_large = large_gemm_indices[-1]
    
    # But large GEMMs might appear at the start of each request's prefill
    # Find the first steady-state decode region (many small GEMMs in a row)
    decode_start = 0
    consecutive_small = 0
    for i, e in enumerate(events):
        if e["stage"] == "GEMM" and e["duration_ns"] < 500_000:  # <500μs
            consecutive_small += 1
            if consecutive_small > 50:  # 50+ small GEMMs = definitely decode
                decode_start = i - consecutive_small + 1
                break
        else:
            consecutive_small = 0

    prefill_events = events[:decode_start]
    decode_events = events[decode_start:]

    return prefill_events, decode_events, decode_start


def segment_decode_tokens(decode_events, num_layers=28):
    """
    Segment decode events into per-token groups.
    
    Each token requires one pass through all layers. The pattern per layer is:
    GEMM (QKV proj) → RoPE → Attn → KV$ → GEMM (out proj) → Norm → GEMM (gate) → GEMM (up) → SiLU → GEMM (down) → Norm
    
    We detect token boundaries by looking for repeating patterns.
    The main signal: the large decode GEMM (128x64, ~200μs) appears once per layer.
    So every `num_layers` of these GEMMs = one token.
    """
    tokens = []
    current_token = []
    main_gemm_count = 0
    
    for e in decode_events:
        current_token.append(e)
        
        # Count the main decode GEMM (the 128x64 one, ~200μs)
        if e["label"] == "GEMM-M" and 100_000 < e["duration_ns"] < 500_000:
            main_gemm_count += 1
            
            if main_gemm_count >= num_layers:
                tokens.append(current_token)
                current_token = []
                main_gemm_count = 0

    # Don't forget the last partial token
    if current_token and len(current_token) > 10:
        tokens.append(current_token)

    return tokens


# ================================================================
# DISPLAY
# ================================================================

def print_token_summary(tokens, max_display=10):
    """Print summary of each token's GPU pipeline."""
    section(f"Token Generation Summary ({len(tokens)} tokens detected)")

    print(f"\n    {C['d']}{'Token':>6}  {'Kernels':>8}  {'GPU Time':>10}  {'GEMM%':>6}  {'Attn%':>6}  {'Other%':>7}  Pipeline{C['r']}")
    print(f"    {C['d']}{'─' * 90}{C['r']}")

    display_count = min(max_display, len(tokens)) if max_display != -1 else len(tokens)

    for i in range(display_count):
        tok = tokens[i]
        total_ns = sum(e["duration_ns"] for e in tok)
        gemm_ns = sum(e["duration_ns"] for e in tok if e["stage"] == "GEMM")
        attn_ns = sum(e["duration_ns"] for e in tok if e["stage"] == "Attn")
        other_ns = total_ns - gemm_ns - attn_ns

        gemm_pct = gemm_ns / total_ns * 100 if total_ns > 0 else 0
        attn_pct = attn_ns / total_ns * 100 if total_ns > 0 else 0
        other_pct = other_ns / total_ns * 100 if total_ns > 0 else 0

        # Build mini pipeline visualization (first 40 kernels)
        pipeline = ""
        for e in tok[:60]:
            color = STAGE_COLORS.get(e["stage"], C["d"])
            char = e["label"][0] if e["label"] else "?"
            pipeline += f"{color}{char}{C['r']}"

        if len(tok) > 60:
            pipeline += f"{C['d']}...+{len(tok)-60}{C['r']}"

        print(f"    {C['v']}{i:>6}{C['r']}"
              f"  {len(tok):>8}"
              f"  {total_ns/1e6:>8.2f}ms"
              f"  {gemm_pct:>5.1f}%"
              f"  {attn_pct:>5.1f}%"
              f"  {other_pct:>6.1f}%"
              f"  {pipeline}")

    if max_display != -1 and len(tokens) > max_display:
        print(f"\n    {C['d']}... {len(tokens) - max_display} more tokens (use --tokens all to show all){C['r']}")

    # Aggregate stats
    all_times = [sum(e["duration_ns"] for e in tok) for tok in tokens]
    if all_times:
        mean_time = sum(all_times) / len(all_times)
        min_time = min(all_times)
        max_time = max(all_times)
        
        section("Decode Token Statistics")
        print(f"    {C['d']}{'Mean GPU time/token':<28}{C['r']}{C['v']}{mean_time/1e6:.3f}{C['r']} ms")
        print(f"    {C['d']}{'Min GPU time/token':<28}{C['r']}{C['v']}{min_time/1e6:.3f}{C['r']} ms")
        print(f"    {C['d']}{'Max GPU time/token':<28}{C['r']}{C['v']}{max_time/1e6:.3f}{C['r']} ms")
        print(f"    {C['d']}{'Tokens detected':<28}{C['r']}{C['v']}{len(tokens)}{C['r']}")
        print(f"    {C['d']}{'Kernels/token (avg)':<28}{C['r']}{C['v']}{sum(len(t) for t in tokens)/len(tokens):.0f}{C['r']}")

        # Implied ITL (GPU time + launch overhead)
        mean_itl_gpu = mean_time / 1e6
        print(f"\n    {C['d']}{'Implied ITL (GPU only)':<28}{C['r']}{C['v']}{mean_itl_gpu:.3f}{C['r']} ms")
        print(f"    {C['d']}{'Note':<28}{C['r']}{C['u']}Actual ITL includes CPU dispatch + scheduling overhead{C['r']}")


def print_token_detail(tokens, token_idx):
    """Print every kernel for a specific token."""
    if token_idx >= len(tokens):
        print(f"{C['e']}Token {token_idx} not found (only {len(tokens)} tokens){C['r']}")
        return

    tok = tokens[token_idx]
    total_ns = sum(e["duration_ns"] for e in tok)

    section(f"Token {token_idx} — Full Kernel Trace ({len(tok)} kernels, {total_ns/1e6:.2f}ms)")

    # Group by layer (approximately — each layer has ~14 kernels)
    print(f"\n    {C['d']}{'#':>4} {'Stage':<6} {'Label':<7} {'Time(μs)':>9} {'Cum(μs)':>9} {'Kernel Name':<55}{C['r']}")
    print(f"    {C['d']}{'─' * 95}{C['r']}")

    cum_ns = 0
    layer_gemm_count = 0
    current_layer = 0

    for j, e in enumerate(tok):
        cum_ns += e["duration_ns"]
        color = STAGE_COLORS.get(e["stage"], C["d"])

        # Detect layer boundaries (main GEMM appears once per layer)
        if e["label"] == "GEMM-M" and 100_000 < e["duration_ns"] < 500_000:
            layer_gemm_count += 1
            if layer_gemm_count > 1:
                current_layer += 1
                print(f"    {C['d']}    {'─── Layer ' + str(current_layer) + ' ───':─<91}{C['r']}")

        short_name = e["name"][:55]
        print(f"    {color}{j:>4} {e['stage']:<6} {e['label']:<7}{C['r']}"
              f" {e['duration_ns']/1e3:>9.1f}"
              f" {cum_ns/1e3:>9.1f}"
              f" {C['d']}{short_name}{C['r']}")

    # Layer summary
    print(f"\n    {C['v']}Total: {total_ns/1e6:.2f}ms across {len(tok)} kernels{C['r']}")


def print_pipeline_legend():
    """Print the pipeline visualization legend."""
    section("Pipeline Legend")
    print(f"    {C['gemm']}G{C['r']} = GEMM (matrix multiply)    "
          f"{C['attn']}A{C['r']} = Flash Attention       "
          f"{C['norm']}F{C['r']}/{C['norm']}N{C['r']} = Normalization")
    print(f"    {C['act']}S{C['r']} = SiLU Activation         "
          f"{C['rope']}R{C['r']} = Rotary Embedding       "
          f"{C['kv']}K{C['r']} = KV Cache Write")
    print(f"    {C['other']}s{C['r']} = Softmax                 "
          f"{C['d']}E{C['r']} = Elementwise            "
          f"{C['d']}?{C['r']} = Other")


def print_prefill_analysis(prefill_events):
    """Analyze the prefill phase."""
    if not prefill_events:
        return

    total_ns = sum(e["duration_ns"] for e in prefill_events)
    section(f"Prefill Phase ({len(prefill_events)} kernels, {total_ns/1e6:.1f}ms)")

    # Group by stage
    stage_times = {}
    for e in prefill_events:
        stage_times.setdefault(e["stage"], {"ns": 0, "count": 0})
        stage_times[e["stage"]]["ns"] += e["duration_ns"]
        stage_times[e["stage"]]["count"] += 1

    sorted_stages = sorted(stage_times.items(), key=lambda x: x[1]["ns"], reverse=True)

    print(f"\n    {C['d']}{'Stage':<12} {'Time(ms)':>10} {'Calls':>7} {'%':>6}{C['r']}")
    print(f"    {C['d']}{'─' * 38}{C['r']}")

    for stage, data in sorted_stages:
        pct = data["ns"] / total_ns * 100 if total_ns > 0 else 0
        color = STAGE_COLORS.get(stage, C["d"])
        print(f"    {color}{stage:<12}{C['r']}"
              f" {data['ns']/1e6:>10.2f}"
              f" {data['count']:>7}"
              f" {pct:>5.1f}%")

    # Show the largest prefill kernels
    print(f"\n    {C['d']}Largest prefill kernels:{C['r']}")
    large = sorted(prefill_events, key=lambda e: e["duration_ns"], reverse=True)[:5]
    for e in large:
        color = STAGE_COLORS.get(e["stage"], C["d"])
        print(f"    {color}{e['label']:<7}{C['r']} {e['duration_ns']/1e6:>8.2f}ms  {C['d']}{e['name'][:60]}{C['r']}")


def print_token_to_token_timing(tokens):
    """Show the wall-clock gap between consecutive tokens (includes CPU overhead)."""
    if len(tokens) < 2:
        return

    section("Token-to-Token Timing (Wall Clock)")
    print(f"    {C['d']}This shows the actual time between the START of each token's GPU work.{C['r']}")
    print(f"    {C['d']}The gap between GPU time and wall time = CPU dispatch + scheduling overhead.{C['r']}\n")

    print(f"    {C['d']}{'Token':>6} {'Wall Δ(ms)':>10} {'GPU(ms)':>9} {'CPU OH(ms)':>10} {'CPU OH%':>8}{C['r']}")
    print(f"    {C['d']}{'─' * 48}{C['r']}")

    wall_deltas = []
    for i in range(1, min(len(tokens), 20)):
        prev_start = tokens[i-1][0]["start_ns"]
        curr_start = tokens[i][0]["start_ns"]
        wall_delta_ns = curr_start - prev_start
        gpu_ns = sum(e["duration_ns"] for e in tokens[i-1])
        cpu_oh_ns = wall_delta_ns - gpu_ns
        cpu_oh_pct = cpu_oh_ns / wall_delta_ns * 100 if wall_delta_ns > 0 else 0

        wall_deltas.append(wall_delta_ns)

        print(f"    {C['v']}{i:>6}{C['r']}"
              f" {wall_delta_ns/1e6:>10.3f}"
              f" {gpu_ns/1e6:>9.3f}"
              f" {cpu_oh_ns/1e6:>10.3f}"
              f" {cpu_oh_pct:>7.1f}%")

    if wall_deltas:
        mean_wall = sum(wall_deltas) / len(wall_deltas)
        print(f"\n    {C['d']}{'Mean wall-clock ITL':<28}{C['r']}{C['v']}{mean_wall/1e6:.3f}{C['r']} ms")
        print(f"    {C['d']}{'This should match Ollama ITL':<28}{C['r']}{C['u']}(~7.17ms for Qwen2.5-7B){C['r']}")


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="PInsight Token-Level GPU Trace")
    parser.add_argument("sqlite_file", help="nsys SQLite export file")
    parser.add_argument("--tokens", default="10",
                        help="Number of decode tokens to show (or 'all')")
    parser.add_argument("--detail-token", type=int, default=None,
                        help="Show full kernel trace for a specific token index")
    parser.add_argument("--layers", type=int, default=28,
                        help="Number of transformer layers in the model (default: 28 for Qwen2.5-7B)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export token timeline to JSON")

    args = parser.parse_args()

    path = Path(args.sqlite_file)
    if not path.exists():
        print(f"{C['e']}File not found: {args.sqlite_file}{C['r']}")
        sys.exit(1)

    banner("PInsight TOKEN-LEVEL GPU TRACE")
    print(f"    {C['d']}Source: {path}{C['r']}")
    print(f"    {C['d']}Model layers: {args.layers}{C['r']}")

    # Extract timeline
    section("Extracting Kernel Timeline")
    events = extract_kernel_timeline(args.sqlite_file)
    print(f"    {C['v']}{len(events):,}{C['r']} kernel events extracted")

    first_ns = events[0]["start_ns"] if events else 0
    last_ns = events[-1]["end_ns"] if events else 0
    total_wall = (last_ns - first_ns) / 1e9
    print(f"    {C['d']}Timeline span: {total_wall:.2f} seconds{C['r']}")

    # Separate prefill and decode
    section("Phase Detection")
    prefill_events, decode_events, boundary = identify_phases(events)
    print(f"    {C['v']}Prefill:{C['r']} {len(prefill_events)} kernels")
    print(f"    {C['v']}Decode:{C['r']}  {len(decode_events)} kernels")

    # Analyze prefill
    print_prefill_analysis(prefill_events)

    # Segment decode into tokens
    section("Token Segmentation")
    tokens = segment_decode_tokens(decode_events, num_layers=args.layers)
    print(f"    {C['v']}{len(tokens)}{C['r']} decode tokens segmented")

    # Display
    print_pipeline_legend()

    max_tokens = -1 if args.tokens == "all" else int(args.tokens)
    print_token_summary(tokens, max_display=max_tokens)

    # Token-to-token wall clock timing
    print_token_to_token_timing(tokens)

    # Detailed view of specific token
    if args.detail_token is not None:
        print_token_detail(tokens, args.detail_token)

    # Export
    if args.export:
        export_data = {
            "source": str(path),
            "total_events": len(events),
            "prefill_events": len(prefill_events),
            "decode_events": len(decode_events),
            "tokens_detected": len(tokens),
            "token_stats": {
                "mean_gpu_ms": sum(sum(e["duration_ns"] for e in t) for t in tokens) / len(tokens) / 1e6 if tokens else 0,
                "per_token": [{
                    "index": i,
                    "kernels": len(t),
                    "gpu_time_ms": round(sum(e["duration_ns"] for e in t) / 1e6, 3),
                    "stages": {},
                } for i, t in enumerate(tokens)],
            },
        }

        # Fill in per-stage breakdown for each token
        for i, t in enumerate(tokens):
            stages = {}
            for e in t:
                stages.setdefault(e["stage"], 0)
                stages[e["stage"]] += e["duration_ns"]
            export_data["token_stats"]["per_token"][i]["stages"] = {
                k: round(v / 1e6, 3) for k, v in stages.items()
            }

        with open(args.export, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\n    {C['ok']}Exported to {args.export}{C['r']}")

    print()


if __name__ == "__main__":
    main()
