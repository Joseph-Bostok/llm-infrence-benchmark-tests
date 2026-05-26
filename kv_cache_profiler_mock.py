#!/usr/bin/env python3
"""
kv_cache_profiler_mock.py — Simulated KV Cache Profiler Results
================================================================
Generates realistic mock results based on our actual A100/Qwen2.5-7B
profiling data from pipeline_profiler.py and nsys_profiler.py.

Use this when aries is unreachable to preview what the profiler captures.
"""

import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Colors ────────────────────────────────────────────────────
C = {
    "h": "\033[1;36m", "ok": "\033[0;32m", "w": "\033[1;33m",
    "e": "\033[0;31m", "d": "\033[0;90m", "v": "\033[0;37m",
    "b": "\033[1;34m", "r": "\033[0m", "m": "\033[1;35m",
}


def banner(text, char="═"):
    w = 70
    print(f"\n{C['h']}{char * w}\n  {text}\n{char * w}{C['r']}")


def section(text):
    print(f"\n{C['b']}── {text} {'─' * max(0, 60 - len(text))}{C['r']}")


def kv(key, val, unit="", indent=4):
    print(f"{' ' * indent}{C['d']}{key}:{C['r']} {C['v']}{val}{C['r']} {C['d']}{unit}{C['r']}")


# ── Model Config (from actual Qwen2.5-7B) ────────────────────

MODEL_CONFIG = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "num_layers": 28,
    "num_attention_heads": 28,
    "num_kv_heads": 4,       # GQA: 7:1 ratio
    "hidden_size": 3584,
    "head_dim": 128,
    "parameters_b": 7.62,
    "weight_bytes": 15_240_000_000 * 2,  # BF16
    "gpu": "NVIDIA A100-PCIE-40GB",
    "dtype": "bfloat16",
    "dtype_bytes": 2,
}


def calc_kv_bytes(seq_len, config=MODEL_CONFIG):
    """Exact KV cache size calculation."""
    return (2 * config["num_layers"] * config["num_kv_heads"]
            * config["head_dim"] * seq_len * config["dtype_bytes"])


def mock_run(max_tokens=128):
    """Simulate a full KV cache profiling run."""

    np.random.seed(42)
    random.seed(42)

    prompt_len = 87  # Realistic for our detailed KV cache prompt
    total_tokens = prompt_len + max_tokens

    banner("KV CACHE DEEP PROFILER (MOCK RUN)", "▓")
    kv("Model", MODEL_CONFIG["model"])
    kv("Max tokens", str(max_tokens))
    kv("Dtype", "bfloat16")
    kv("Device", "cuda:0")
    kv("GPU", MODEL_CONFIG["gpu"])
    kv("GPU Memory", "33.2 / 40.0 GB free")
    print(f"\n    {C['w']}⚠  MOCK MODE — results simulated from real A100 baselines{C['r']}")

    # ── Load ──────────────────────────────────────────────────
    section("Loading Model")
    kv("Load time", "12.3", "s")
    kv("Parameters", "7.62", "B")
    kv("Config", "28L / 28Q / 4KV / dim 128")
    kv("GPU after load", "14578", "MB")

    # ── Warmup ────────────────────────────────────────────────
    section("Warmup (1 short generation)")
    print(f"    {C['ok']}Warmup complete{C['r']}")

    # ── Tokenize ──────────────────────────────────────────────
    section("Tokenizing Prompt")
    kv("Prompt tokens", prompt_len)

    # ── Prefill ───────────────────────────────────────────────
    section("Prefill (with attention capture)")

    prefill_kv_theoretical = calc_kv_bytes(prompt_len)
    kv("Prefill time", "38.7", "ms")
    kv("Memory after prefill", "14612", "MB")
    kv("KV cache (measured)", "34.2", "MB")
    kv("KV cache (theoretical)", f"{prefill_kv_theoretical / 1e6:.1f}", "MB")

    # ── Decode Loop ───────────────────────────────────────────
    section(f"Decode Loop ({max_tokens} tokens, tracking KV cache)")

    # Generate realistic per-token metrics
    # Base decode latency: ~35ms with eager attention (not flash)
    # Slight upward trend as KV cache grows
    base_latency = 34.5  # ms (eager attention is slower than flash)
    latency_growth_per_token = 0.008  # ms per token of sequence growth

    per_token = []
    latencies = []

    for step in range(max_tokens):
        seq_len = prompt_len + step
        latency = base_latency + latency_growth_per_token * step + np.random.normal(0, 0.3)
        latency = max(latency, base_latency * 0.95)

        kv_bytes = calc_kv_bytes(seq_len)

        token_data = {
            "step": step,
            "seq_len": seq_len,
            "kv_cache_bytes_measured": int(kv_bytes * 1.35),  # PyTorch overhead
            "kv_cache_bytes_theoretical": kv_bytes,
            "kv_cache_mb": round(kv_bytes * 1.35 / (1024**2), 2),
            "decode_latency_ms": round(latency, 3),
            "gpu_mem_mb": round(14612 + kv_bytes * 1.35 / (1024**2), 1),
        }

        per_token.append(token_data)
        latencies.append(latency)

        if (step + 1) % 32 == 0 or step == 0:
            print(f"    {C['d']}Step {step+1:>4}/{max_tokens}: "
                  f"latency={latency:.1f}ms  "
                  f"KV={token_data['kv_cache_mb']:.1f}MB  "
                  f"seq_len={seq_len}"
                  f"{C['r']}")

    # ── Attention Analysis ────────────────────────────────────
    section("Attention Analysis")

    # Generate realistic attention importance distribution
    # Key patterns:
    # 1. First few tokens get disproportionate attention (sinks)
    # 2. A few content tokens are heavy hitters
    # 3. Most tokens get very low attention
    importance = np.zeros(total_tokens)

    # Attention sinks: first 4 tokens get massive attention
    importance[0] = 0.95   # BOS / first token = biggest sink
    importance[1] = 0.42
    importance[2] = 0.28
    importance[3] = 0.19

    # Content heavy hitters: scattered throughout prompt
    heavy_hitter_positions = [8, 15, 23, 31, 45, 52, 67, 78]
    for pos in heavy_hitter_positions:
        if pos < total_tokens:
            importance[pos] = 0.15 + np.random.uniform(0.1, 0.35)

    # Regular tokens: low but non-zero
    for i in range(total_tokens):
        if importance[i] == 0:
            importance[i] = np.random.exponential(0.03)

    # Recent tokens get modest attention (recency bias)
    for i in range(max(0, total_tokens - 20), total_tokens):
        importance[i] = max(importance[i], 0.05 + np.random.uniform(0, 0.08))

    # Normalize to [0, 1]
    importance_norm = importance / importance.max()

    # Attention sink share
    sink_share = importance[:4].sum() / importance.sum()

    # Heavy hitters
    mean_imp = importance.mean()
    std_imp = importance.std()
    threshold = mean_imp + 2 * std_imp
    heavy_hitters = importance > threshold
    n_hh = int(heavy_hitters.sum())
    hh_pct = n_hh / total_tokens * 100

    # Per-layer entropy (simulated)
    # Early layers: low entropy (selective, focused attention)
    # Middle layers: medium entropy
    # Late layers: high entropy (diffuse attention)
    layer_entropy = []
    for l in range(MODEL_CONFIG["num_layers"]):
        if l < 7:
            e = 3.0 + np.random.uniform(0, 0.5)   # selective
        elif l < 21:
            e = 4.2 + np.random.uniform(0, 0.6)   # medium
        else:
            e = 5.0 + np.random.uniform(0, 0.4)   # diffuse
        layer_entropy.append(round(e, 2))

    most_selective = int(np.argmin(layer_entropy))
    least_selective = int(np.argmax(layer_entropy))
    top_important = [int(i) for i in np.argsort(importance)[-10:][::-1]]

    kv("Total tokens", total_tokens)
    kv("Attention sink (first 4 tokens)", f"{sink_share:.1%}")
    kv("Heavy hitters (H2O-style)", f"{n_hh} ({hh_pct:.1f}%)")
    kv("Most selective layer", f"Layer {most_selective}")
    kv("Least selective layer", f"Layer {least_selective}")
    kv("Top important positions", top_important)

    # ── Bandwidth Split ───────────────────────────────────────
    section("Bandwidth Split Analysis")

    model_bytes = MODEL_CONFIG["parameters_b"] * 1e9 * MODEL_CONFIG["dtype_bytes"]
    test_seq_lens = [64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    bandwidth_points = []
    crossover_seq = None

    print(f"\n    {C['d']}{'Seq Len':>10} {'KV Cache':>12} {'Weights':>12} {'KV/Weight':>10} {'KV % BW':>10}{C['r']}")
    print(f"    {C['d']}{'─' * 58}{C['r']}")

    for seq_l in test_seq_lens:
        kv_bytes = calc_kv_bytes(seq_l)
        ratio = kv_bytes / model_bytes
        kv_pct = ratio / (1 + ratio) * 100

        pt = {
            "seq_len": seq_l,
            "kv_mb": round(kv_bytes / 1e6, 1),
            "kv_gb": round(kv_bytes / 1e9, 2),
            "weight_gb": round(model_bytes / 1e9, 2),
            "kv_to_weight_ratio": round(ratio, 4),
            "kv_pct_of_total_bandwidth": round(kv_pct, 1),
        }
        bandwidth_points.append(pt)

        if crossover_seq is None and ratio >= 0.10:
            crossover_seq = seq_l

        marker = " ←" if seq_l == crossover_seq else ""
        color = C['w'] if ratio >= 0.10 else C['v']
        print(f"    {color}{seq_l:>10,} {pt['kv_mb']:>10.1f} MB "
              f"{pt['weight_gb']:>10.2f} GB {ratio:>9.2%} "
              f"{kv_pct:>8.1f}%{marker}{C['r']}")

    if crossover_seq:
        kv("KV reaches 10% of weight BW at", f"{crossover_seq:,} tokens")

    # ── Decode Latency Analysis ───────────────────────────────
    section("Decode Latency vs Sequence Position")

    lat_arr = np.array(latencies)
    p50 = float(np.percentile(lat_arr, 50))
    p90 = float(np.percentile(lat_arr, 90))
    p99 = float(np.percentile(lat_arr, 99))

    q_size = max(len(lat_arr) // 4, 1)
    first_q = lat_arr[:q_size]
    last_q = lat_arr[-q_size:]
    growth = (last_q.mean() / first_q.mean() - 1) * 100

    kv("P50 latency", f"{p50:.2f}", "ms")
    kv("P90 latency", f"{p90:.2f}", "ms")
    kv("P99 latency", f"{p99:.2f}", "ms")
    kv("First 25% mean", f"{first_q.mean():.2f}", "ms")
    kv("Last 25% mean", f"{last_q.mean():.2f}", "ms")
    kv("Latency growth", f"{growth:+.1f}%",
       "(positive = slowing down with longer seq)")
    kv("Throughput", f"{len(latencies) / (lat_arr.sum() / 1000):.1f}", "tok/s")

    # ── Eviction Simulation ───────────────────────────────────
    section("Eviction Simulation")

    budgets = [0.10, 0.25, 0.50, 0.75, 0.90]
    eviction_results = []

    print(f"\n    {C['d']}{'Budget':>8} {'Tokens':>8} {'Attn Coverage':>15} "
          f"{'KV Before':>10} {'KV After':>10} {'Saved':>10} {'Ratio':>8}{C['r']}")
    print(f"    {C['d']}{'─' * 73}{C['r']}")

    for budget in budgets:
        tokens_to_keep = max(1, int(total_tokens * budget))

        # Simulate H2O eviction: keep sinks + top-K by importance
        sink_positions = set(range(min(4, total_tokens)))
        remaining = tokens_to_keep - len(sink_positions)
        if remaining > 0:
            non_sink = importance.copy()
            for s in sink_positions:
                non_sink[s] = -1
            selected = set(np.argsort(non_sink)[-remaining:])
            selected.update(sink_positions)
        else:
            selected = sink_positions

        coverage = importance[list(selected)].sum() / importance.sum()

        orig_kv = calc_kv_bytes(total_tokens)
        kept_kv = calc_kv_bytes(tokens_to_keep)

        sim = {
            "budget_pct": int(budget * 100),
            "tokens_kept": tokens_to_keep,
            "h2o_attention_coverage": round(float(coverage), 4),
            "original_kv_mb": round(orig_kv / 1e6, 1),
            "evicted_kv_mb": round(kept_kv / 1e6, 1),
            "memory_saved_mb": round((orig_kv - kept_kv) / 1e6, 1),
            "compression_ratio": round(orig_kv / max(kept_kv, 1), 2),
        }
        eviction_results.append(sim)

        cov_color = C['ok'] if coverage > 0.95 else \
                    C['w'] if coverage > 0.85 else C['e']
        print(f"    {C['v']}{sim['budget_pct']:>7}% {sim['tokens_kept']:>7}  "
              f"{cov_color}{coverage:>14.1%}{C['r']}  "
              f"{C['d']}{sim['original_kv_mb']:>8.1f}MB {sim['evicted_kv_mb']:>8.1f}MB "
              f"{sim['memory_saved_mb']:>8.1f}MB {sim['compression_ratio']:>7.1f}×{C['r']}")

    # ── Summary ───────────────────────────────────────────────
    section("Summary")

    ev50 = eviction_results[2]  # 50% budget
    print(f"""
    {C['h']}KV Cache Profile Results (Qwen2.5-7B on A100):{C['r']}

    {C['v']}Model:{C['r']}        28L / 4 KV heads (GQA 7:1) / dim 128
    {C['v']}Prompt:{C['r']}       {prompt_len} tokens
    {C['v']}Generated:{C['r']}    {max_tokens} tokens
    {C['v']}Final KV:{C['r']}     {per_token[-1]['kv_cache_mb']:.1f} MB (measured) / {calc_kv_bytes(total_tokens)/1e6:.1f} MB (theoretical)

    {C['m']}Attention Insights:{C['r']}
      Sink tokens (first 4) absorb {sink_share:.1%} of total attention
      Only {hh_pct:.1f}% of tokens ({n_hh}/{total_tokens}) are heavy hitters
      → {100 - hh_pct:.0f}% of tokens could potentially be evicted

    {C['m']}Bandwidth:{C['r']}
      KV cache becomes >10% of bandwidth at seq_len ~{crossover_seq:,}
      At 128K tokens: KV reads ≈ 50% of weight reads

    {C['m']}Latency:{C['r']}
      Decode: {p50:.1f} ms/tok (P50), {p99:.1f} ms/tok (P99)
      Growth: {growth:+.1f}% over {max_tokens} tokens (minimal at this length)

    {C['m']}Eviction Simulation (50% budget):{C['r']}
      Keeps {ev50['h2o_attention_coverage']:.1%} of attention mass
      Saves {ev50['memory_saved_mb']:.1f} MB ({ev50['compression_ratio']:.1f}× compression)

    {C['m']}Layer Selectivity:{C['r']}
      Most selective:  Layer {most_selective} (entropy: {layer_entropy[most_selective]:.2f})
      Least selective: Layer {least_selective} (entropy: {layer_entropy[least_selective]:.2f})
      → Early layers need MORE KV budget, late layers can be compressed harder
    """)

    # ── ASCII Charts ──────────────────────────────────────────

    # Decode latency curve
    section("Decode Latency Curve (per-token)")
    _ascii_chart(latencies, width=60, height=10,
                 xlabel="Token Position", ylabel="Latency (ms)")

    # Token importance
    section("Token Importance (H2O-style cumulative attention)")
    _ascii_chart(list(importance_norm[:min(total_tokens, 80)]), width=60, height=8,
                 xlabel="Token Position", ylabel="Importance")

    # Layer entropy
    section("Per-Layer Attention Entropy")
    _ascii_chart(layer_entropy, width=28, height=8,
                 xlabel="Layer Index", ylabel="Entropy (bits)")

    # KV cache growth
    section("KV Cache Growth During Decode")
    kv_sizes = [t["kv_cache_mb"] for t in per_token]
    _ascii_chart(kv_sizes, width=60, height=8,
                 xlabel="Decode Step", ylabel="KV Size (MB)")

    # ── Save Results ──────────────────────────────────────────
    output_dir = Path("./results/kv_cache/mock_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "mock_run": True,
        "timestamp": datetime.now().isoformat(),
        "model_config": MODEL_CONFIG,
        "prompt_tokens": prompt_len,
        "generated_tokens": max_tokens,
        "total_tokens": total_tokens,
        "per_token": per_token,
        "attention_data": {
            "attention_sink_share": round(float(sink_share), 4),
            "n_heavy_hitters": n_hh,
            "heavy_hitter_pct": round(float(hh_pct), 1),
            "most_selective_layer": most_selective,
            "least_selective_layer": least_selective,
            "layer_entropy_avg": layer_entropy,
            "top_10_important_positions": top_important,
            "importance_scores": [round(float(s), 4) for s in importance_norm],
        },
        "bandwidth_analysis": {
            "model_weight_gb": round(model_bytes / 1e9, 2),
            "crossover_10pct_seq_len": crossover_seq,
            "points": bandwidth_points,
        },
        "latency_analysis": {
            "p50_ms": round(p50, 2),
            "p90_ms": round(p90, 2),
            "p99_ms": round(p99, 2),
            "first_quarter_mean_ms": round(float(first_q.mean()), 2),
            "last_quarter_mean_ms": round(float(last_q.mean()), 2),
            "latency_growth_pct": round(float(growth), 1),
            "tokens_per_second": round(len(latencies) / (lat_arr.sum() / 1000), 1),
        },
        "eviction_simulation": eviction_results,
    }

    result_path = output_dir / "kv_cache_results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    section("Results Saved")
    kv("Output", str(result_path))

    banner("MOCK PROFILING COMPLETE", "▓")

    return result


def _ascii_chart(values, width=60, height=10, xlabel="", ylabel=""):
    """Minimal ASCII bar chart for terminal visualization."""
    n = len(values)
    v_min = min(values)
    v_max = max(values)
    v_range = v_max - v_min if v_max != v_min else 1

    # Bucket values
    bins = [[] for _ in range(width)]
    for i, v in enumerate(values):
        bin_idx = min(int(i / n * width), width - 1)
        bins[bin_idx].append(v)

    bin_avgs = [sum(b)/len(b) if b else 0 for b in bins]

    print(f"    {C['d']}{ylabel}{C['r']}")
    for row in range(height - 1, -1, -1):
        threshold = v_min + (row / max(height - 1, 1)) * v_range
        line = ""
        for b_avg in bin_avgs:
            if b_avg >= threshold:
                line += "█"
            elif b_avg >= threshold - v_range / height * 0.5:
                line += "▄"
            else:
                line += " "
        if row == height - 1:
            label = f"{v_max:>8.1f}"
        elif row == 0:
            label = f"{v_min:>8.1f}"
        else:
            label = " " * 8
        print(f"    {C['d']}{label}{C['r']} │{C['m']}{line}{C['r']}│")

    print(f"    {' ' * 8} └{'─' * width}┘")
    print(f"    {' ' * 8}  {xlabel}")
    print(f"    {C['d']}  n={n}  min={v_min:.2f}  max={v_max:.2f}  "
          f"mean={sum(values)/len(values):.2f}{C['r']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mock KV Cache Profiler")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    mock_run(max_tokens=args.max_tokens)
