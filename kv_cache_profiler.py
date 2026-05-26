#!/usr/bin/env python3
"""
kv_cache_profiler.py — KV Cache Deep Profiler
==============================================
Profiles KV cache behavior during LLM inference at per-token,
per-layer granularity. Captures:

  1. KV cache memory growth per decode step
  2. Per-token decode latency vs sequence position (latency cliff)
  3. Token importance scoring (H2O-style cumulative attention)
  4. Per-head attention distribution (SnapKV-style analysis)
  5. Attention sink detection (StreamingLLM pattern)
  6. Bandwidth split analysis (weight reads vs KV cache reads)
  7. Simulated eviction at multiple budgets

Uses bare HuggingFace transformers with output_attentions=True for
full visibility into KV cache and attention patterns.

Usage:
  python3 kv_cache_profiler.py --model Qwen/Qwen2.5-7B-Instruct
  python3 kv_cache_profiler.py --model Qwen/Qwen2.5-7B-Instruct --max-tokens 512
  python3 kv_cache_profiler.py --model Qwen/Qwen2.5-7B-Instruct --prompt "long prompt here..."

Author: PInsight Benchmark Suite
"""

import argparse
import gc
import json
import os
import sys
import time
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


# ── GPU Helpers ───────────────────────────────────────────────

def gpu_mem_mb():
    import torch
    return torch.cuda.memory_allocated() / (1024 ** 2)


def cuda_event_pair():
    import torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def event_elapsed_ms(start, end):
    import torch
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ── Model Info ────────────────────────────────────────────────

def get_model_kv_config(model):
    """Extract KV cache configuration from model config."""
    config = model.config
    info = {}
    info["num_layers"] = getattr(config, "num_hidden_layers", None)
    info["num_attention_heads"] = getattr(config, "num_attention_heads", None)
    info["num_kv_heads"] = getattr(config, "num_key_value_heads",
                                   info["num_attention_heads"])
    info["hidden_size"] = getattr(config, "hidden_size", None)
    info["head_dim"] = info["hidden_size"] // info["num_attention_heads"] \
        if info["hidden_size"] and info["num_attention_heads"] else None
    return info


def calc_kv_cache_bytes(model_info, seq_len, dtype_bytes=2):
    """Calculate theoretical KV cache size in bytes."""
    # KV cache = 2 (K+V) × num_layers × num_kv_heads × head_dim × seq_len × dtype
    return (2 * model_info["num_layers"] * model_info["num_kv_heads"]
            * model_info["head_dim"] * seq_len * dtype_bytes)


# ── Prompts ───────────────────────────────────────────────────

DEFAULT_PROMPT = (
    "Explain in detail how key-value caching works in transformer-based "
    "language models during autoregressive generation. Cover the following "
    "aspects: why the KV cache exists, how it grows during generation, "
    "what happens to memory bandwidth as the cache gets larger, and what "
    "optimization techniques exist to manage KV cache size. Include specific "
    "examples of how attention patterns determine which cached tokens are "
    "most important for generating accurate outputs."
)


# ── KV Cache Profiling Core ──────────────────────────────────

def profile_kv_cache(model, tokenizer, prompt, max_new_tokens, device):
    """
    Run inference with full KV cache + attention weight tracking.
    Returns comprehensive per-token, per-layer analysis.
    """
    import torch

    result = {
        "prompt": prompt[:120],
        "max_new_tokens": max_new_tokens,
        "model_config": {},
        "per_token": [],       # per decode step metrics
        "attention_data": {},  # aggregated attention analysis
        "bandwidth_analysis": {},
        "eviction_simulation": {},
    }

    model_info = get_model_kv_config(model)
    result["model_config"] = model_info
    num_layers = model_info["num_layers"]
    num_kv_heads = model_info["num_kv_heads"]
    num_attn_heads = model_info["num_attention_heads"]
    head_dim = model_info["head_dim"]

    dtype_bytes = 2  # bfloat16

    section("Tokenizing Prompt")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    prompt_len = input_ids.shape[1]
    kv("Prompt tokens", prompt_len)

    input_ids_gpu = input_ids.to(device)

    # ── Prefill ───────────────────────────────────────────────
    section("Prefill (with attention capture)")
    mem_before = gpu_mem_mb()

    s_pf, e_pf = cuda_event_pair()
    s_pf.record()
    with torch.no_grad():
        outputs = model(
            input_ids_gpu,
            use_cache=True,
            output_attentions=True,
        )
    e_pf.record()
    prefill_ms = event_elapsed_ms(s_pf, e_pf)

    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]

    # Capture prefill attention for observation-window analysis (SnapKV-style)
    # outputs.attentions: tuple of (num_layers,) tensors
    # Each shape: (batch, num_heads, seq_len, seq_len)
    prefill_attention = []
    if outputs.attentions is not None:
        for layer_attn in outputs.attentions:
            # Take last token's attention as the observation window
            # shape: (num_heads, seq_len)
            last_tok_attn = layer_attn[0, :, -1, :].cpu().float().numpy()
            prefill_attention.append(last_tok_attn)

    del outputs.attentions  # free memory
    mem_after_prefill = gpu_mem_mb()

    kv("Prefill time", f"{prefill_ms:.1f}", "ms")
    kv("Memory after prefill", f"{mem_after_prefill:.0f}", "MB")
    kv("KV cache (measured)", f"{mem_after_prefill - mem_before:.1f}", "MB")
    kv("KV cache (theoretical)", f"{calc_kv_cache_bytes(model_info, prompt_len, dtype_bytes) / 1e6:.1f}", "MB")

    # ── Decode Loop with KV Tracking ──────────────────────────
    section(f"Decode Loop ({max_new_tokens} tokens, tracking KV cache)")

    # Accumulators for attention analysis
    # cumulative_importance: sum of attention received by each token position
    # shape: (seq_len,) — grows each step
    cumulative_importance = np.zeros(prompt_len, dtype=np.float64)

    # Per-head importance tracking (for SnapKV-style analysis)
    # Will accumulate per-head attention distributions
    per_head_cumulative = np.zeros((num_layers, num_attn_heads, prompt_len),
                                   dtype=np.float64)

    # Track per-layer attention entropy (how spread out is attention?)
    layer_entropy_accum = np.zeros(num_layers, dtype=np.float64)

    generated_ids = []
    eos_id = tokenizer.eos_token_id
    decode_steps = 0

    for step in range(max_new_tokens):
        token_data = {"step": step}
        current_seq_len = prompt_len + step

        # ── Sample next token ─────────────────────────────────
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        token_id = next_token.item()
        generated_ids.append(token_id)

        if token_id == eos_id:
            token_data["eos"] = True
            result["per_token"].append(token_data)
            break

        # ── Measure KV cache size ─────────────────────────────
        kv_bytes_measured = 0
        if past_key_values is not None:
            for layer_kv in past_key_values:
                # Each layer_kv is a tuple of (key, value) tensors
                for t in layer_kv:
                    kv_bytes_measured += t.numel() * t.element_size()

        kv_bytes_theoretical = calc_kv_cache_bytes(
            model_info, current_seq_len, dtype_bytes
        )

        token_data["kv_cache_bytes_measured"] = kv_bytes_measured
        token_data["kv_cache_bytes_theoretical"] = kv_bytes_theoretical
        token_data["kv_cache_mb"] = round(kv_bytes_measured / (1024**2), 2)
        token_data["seq_len"] = current_seq_len

        # ── Forward pass with attention capture ───────────────
        s_fwd, e_fwd = cuda_event_pair()
        s_fwd.record()
        with torch.no_grad():
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
        e_fwd.record()
        fwd_ms = event_elapsed_ms(s_fwd, e_fwd)

        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        token_data["decode_latency_ms"] = round(fwd_ms, 3)
        token_data["gpu_mem_mb"] = round(gpu_mem_mb(), 1)

        # ── Process attention weights ─────────────────────────
        if outputs.attentions is not None:
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                # shape: (1, num_heads, 1, current_seq_len+1)
                # The query (dim 2) is length 1 (single decode token)
                # The key (dim 3) is current_seq_len+1 (all cached + current)
                attn = layer_attn[0, :, 0, :].cpu().float().numpy()
                # attn shape: (num_heads, current_seq_len+1)

                attn_len = attn.shape[1]
                if attn_len > len(cumulative_importance):
                    # Grow the arrays to accommodate new tokens
                    new_size = attn_len
                    old_size = len(cumulative_importance)
                    cumulative_importance = np.pad(
                        cumulative_importance, (0, new_size - old_size)
                    )
                    per_head_cumulative = np.pad(
                        per_head_cumulative,
                        ((0, 0), (0, 0), (0, new_size - old_size))
                    )

                # Accumulate attention scores
                # Sum across heads for global importance
                token_importance = attn.sum(axis=0)  # shape: (current_seq_len+1,)
                cumulative_importance[:attn_len] += token_importance

                # Per-head accumulation
                per_head_cumulative[layer_idx, :, :attn_len] += attn

                # Entropy (how spread out is attention for this layer?)
                # Higher entropy = more uniform attention = less selective
                attn_mean = attn.mean(axis=0)  # avg across heads
                attn_mean = attn_mean + 1e-10  # avoid log(0)
                entropy = -np.sum(attn_mean * np.log2(attn_mean))
                layer_entropy_accum[layer_idx] += entropy

            del outputs.attentions

        result["per_token"].append(token_data)
        decode_steps += 1

        # Progress
        if (step + 1) % 32 == 0 or step == 0:
            print(f"    {C['d']}Step {step+1:>4}/{max_new_tokens}: "
                  f"latency={fwd_ms:.1f}ms  "
                  f"KV={token_data['kv_cache_mb']:.1f}MB  "
                  f"seq_len={current_seq_len}"
                  f"{C['r']}")

    n_generated = len(generated_ids)
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ── Attention Analysis ────────────────────────────────────
    section("Attention Analysis")

    total_tokens = prompt_len + n_generated
    importance = cumulative_importance[:total_tokens]

    # Normalize importance to [0, 1]
    if importance.max() > 0:
        importance_norm = importance / importance.max()
    else:
        importance_norm = importance

    # Attention sinks: how much attention goes to first 4 tokens?
    sink_tokens = min(4, len(importance))
    sink_attention_share = float(importance[:sink_tokens].sum() / max(importance.sum(), 1e-10))

    # Find heavy hitters (H2O-style): tokens receiving > mean + 2*std attention
    mean_imp = importance.mean()
    std_imp = importance.std()
    heavy_hitter_threshold = mean_imp + 2 * std_imp
    heavy_hitter_mask = importance > heavy_hitter_threshold
    n_heavy_hitters = int(heavy_hitter_mask.sum())
    heavy_hitter_pct = n_heavy_hitters / max(total_tokens, 1) * 100

    # Per-layer entropy (averaged over decode steps)
    layer_entropy_avg = layer_entropy_accum / max(decode_steps, 1)

    # Identify most and least selective layers
    most_selective_layer = int(np.argmin(layer_entropy_avg))
    least_selective_layer = int(np.argmax(layer_entropy_avg))

    result["attention_data"] = {
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_len,
        "generated_tokens": n_generated,
        "attention_sink_share": round(sink_attention_share, 4),
        "n_heavy_hitters": n_heavy_hitters,
        "heavy_hitter_pct": round(heavy_hitter_pct, 1),
        "heavy_hitter_threshold": round(float(heavy_hitter_threshold), 4),
        "most_selective_layer": most_selective_layer,
        "least_selective_layer": least_selective_layer,
        "layer_entropy_avg": [round(float(e), 2) for e in layer_entropy_avg],
        "top_10_important_positions": [
            int(i) for i in np.argsort(importance)[-10:][::-1]
        ],
        "importance_scores": [round(float(s), 4) for s in importance_norm],
    }

    kv("Total tokens", total_tokens)
    kv("Attention sink (first 4 tokens)", f"{sink_attention_share:.1%}")
    kv("Heavy hitters (H2O-style)", f"{n_heavy_hitters} ({heavy_hitter_pct:.1f}%)")
    kv("Most selective layer", f"Layer {most_selective_layer}")
    kv("Least selective layer", f"Layer {least_selective_layer}")
    kv("Top important positions", result["attention_data"]["top_10_important_positions"])

    # ── Bandwidth Split Analysis ──────────────────────────────
    section("Bandwidth Split Analysis")

    model_params = sum(p.numel() for p in model.parameters())
    model_bytes = model_params * dtype_bytes

    # At different sequence lengths, compute KV vs weight byte ratio
    bandwidth_points = []
    test_seq_lens = [64, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    for seq_l in test_seq_lens:
        kv_bytes = calc_kv_cache_bytes(model_info, seq_l, dtype_bytes)
        ratio = kv_bytes / model_bytes
        bandwidth_points.append({
            "seq_len": seq_l,
            "kv_bytes": kv_bytes,
            "kv_mb": round(kv_bytes / 1e6, 1),
            "kv_gb": round(kv_bytes / 1e9, 2),
            "weight_bytes": model_bytes,
            "weight_gb": round(model_bytes / 1e9, 2),
            "kv_to_weight_ratio": round(ratio, 4),
            "kv_pct_of_total_bandwidth": round(ratio / (1 + ratio) * 100, 1),
        })

    # Find crossover point (where KV > 10% of weight reads)
    crossover_seq = None
    for pt in bandwidth_points:
        if pt["kv_to_weight_ratio"] >= 0.10:
            crossover_seq = pt["seq_len"]
            break

    result["bandwidth_analysis"] = {
        "model_weight_bytes": model_bytes,
        "model_weight_gb": round(model_bytes / 1e9, 2),
        "points": bandwidth_points,
        "crossover_10pct_seq_len": crossover_seq,
    }

    print(f"\n    {C['d']}{'Seq Len':>10} {'KV Cache':>12} {'Weights':>12} {'KV/Weight':>10} {'KV % BW':>10}{C['r']}")
    print(f"    {C['d']}{'─' * 58}{C['r']}")
    for pt in bandwidth_points:
        marker = " ←" if pt["seq_len"] == crossover_seq else ""
        color = C['w'] if pt["kv_to_weight_ratio"] >= 0.10 else C['v']
        print(f"    {color}{pt['seq_len']:>10,} {pt['kv_mb']:>10.1f} MB "
              f"{pt['weight_gb']:>10.2f} GB {pt['kv_to_weight_ratio']:>9.2%} "
              f"{pt['kv_pct_of_total_bandwidth']:>8.1f}%{marker}{C['r']}")

    if crossover_seq:
        kv("KV reaches 10% of weight BW at", f"{crossover_seq:,} tokens")

    # ── Decode Latency vs Sequence Position ───────────────────
    section("Decode Latency vs Sequence Position")

    latencies = [t["decode_latency_ms"] for t in result["per_token"]
                 if "decode_latency_ms" in t]

    if latencies:
        lat_arr = np.array(latencies)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p99 = float(np.percentile(lat_arr, 99))

        # Check for latency increase: compare first quarter vs last quarter
        q1 = lat_arr[:len(lat_arr)//4]
        q4 = lat_arr[-(len(lat_arr)//4):] if len(lat_arr) >= 4 else lat_arr

        first_q_mean = float(q1.mean()) if len(q1) > 0 else 0
        last_q_mean = float(q4.mean()) if len(q4) > 0 else 0
        latency_growth = (last_q_mean / first_q_mean - 1) * 100 if first_q_mean > 0 else 0

        result["latency_analysis"] = {
            "p50_ms": round(p50, 2),
            "p90_ms": round(p90, 2),
            "p99_ms": round(p99, 2),
            "min_ms": round(float(lat_arr.min()), 2),
            "max_ms": round(float(lat_arr.max()), 2),
            "first_quarter_mean_ms": round(first_q_mean, 2),
            "last_quarter_mean_ms": round(last_q_mean, 2),
            "latency_growth_pct": round(latency_growth, 1),
            "tokens_per_second": round(len(latencies) / (lat_arr.sum() / 1000), 1),
        }

        kv("P50 latency", f"{p50:.2f}", "ms")
        kv("P90 latency", f"{p90:.2f}", "ms")
        kv("P99 latency", f"{p99:.2f}", "ms")
        kv("First 25% mean", f"{first_q_mean:.2f}", "ms")
        kv("Last 25% mean", f"{last_q_mean:.2f}", "ms")
        kv("Latency growth", f"{latency_growth:+.1f}%",
           "(positive = slowing down with longer seq)")
        kv("Throughput", f"{result['latency_analysis']['tokens_per_second']:.1f}", "tok/s")

    # ── Eviction Simulation ───────────────────────────────────
    section("Eviction Simulation")

    budgets = [0.10, 0.25, 0.50, 0.75, 0.90]
    eviction_results = []

    for budget in budgets:
        tokens_to_keep = max(1, int(total_tokens * budget))
        sim = {"budget_pct": int(budget * 100), "tokens_kept": tokens_to_keep}

        # H2O-style: keep top-K by cumulative importance + always keep sinks
        sink_positions = set(range(min(4, total_tokens)))
        remaining_budget = tokens_to_keep - len(sink_positions)

        if remaining_budget > 0:
            # Get importance of non-sink tokens
            non_sink_importance = importance.copy()
            for s in sink_positions:
                non_sink_importance[s] = -1  # exclude sinks from ranking

            h2o_selected = set(np.argsort(non_sink_importance)[-remaining_budget:])
            h2o_selected.update(sink_positions)
        else:
            h2o_selected = sink_positions

        # Coverage: what % of total attention mass is captured?
        h2o_coverage = float(importance[list(h2o_selected)].sum()
                            / max(importance.sum(), 1e-10))

        # Memory savings
        original_kv_bytes = calc_kv_cache_bytes(model_info, total_tokens, dtype_bytes)
        evicted_kv_bytes = calc_kv_cache_bytes(model_info, tokens_to_keep, dtype_bytes)

        sim["h2o_attention_coverage"] = round(h2o_coverage, 4)
        sim["original_kv_mb"] = round(original_kv_bytes / 1e6, 1)
        sim["evicted_kv_mb"] = round(evicted_kv_bytes / 1e6, 1)
        sim["memory_saved_mb"] = round((original_kv_bytes - evicted_kv_bytes) / 1e6, 1)
        sim["compression_ratio"] = round(original_kv_bytes / max(evicted_kv_bytes, 1), 2)

        eviction_results.append(sim)

    result["eviction_simulation"] = eviction_results

    print(f"\n    {C['d']}{'Budget':>8} {'Tokens':>8} {'Attn Coverage':>15} "
          f"{'KV Before':>10} {'KV After':>10} {'Saved':>10} {'Ratio':>8}{C['r']}")
    print(f"    {C['d']}{'─' * 73}{C['r']}")
    for sim in eviction_results:
        cov_color = C['ok'] if sim["h2o_attention_coverage"] > 0.95 else \
                    C['w'] if sim["h2o_attention_coverage"] > 0.85 else C['e']
        print(f"    {C['v']}{sim['budget_pct']:>7}% {sim['tokens_kept']:>7}  "
              f"{cov_color}{sim['h2o_attention_coverage']:>14.1%}{C['r']}  "
              f"{C['d']}{sim['original_kv_mb']:>8.1f}MB {sim['evicted_kv_mb']:>8.1f}MB "
              f"{sim['memory_saved_mb']:>8.1f}MB {sim['compression_ratio']:>7.1f}×{C['r']}")

    # ── Summary ───────────────────────────────────────────────
    section("Summary")

    print(f"""
    {C['h']}KV Cache Profile Results:{C['r']}

    {C['v']}Model:{C['r']}        {model_info['num_layers']}L / {model_info['num_kv_heads']} KV heads / dim {model_info['head_dim']}
    {C['v']}Prompt:{C['r']}       {prompt_len} tokens
    {C['v']}Generated:{C['r']}    {n_generated} tokens
    {C['v']}Final KV:{C['r']}     {result['per_token'][-1].get('kv_cache_mb', 'N/A')} MB

    {C['m']}Attention Insights:{C['r']}
      Sink tokens absorb {sink_attention_share:.1%} of total attention
      Only {heavy_hitter_pct:.1f}% of tokens are heavy hitters
      → {100 - heavy_hitter_pct:.0f}% of tokens could potentially be evicted

    {C['m']}Bandwidth:{C['r']}
      KV cache becomes >10% of bandwidth at seq_len={crossover_seq or 'N/A'}

    {C['m']}Latency:{C['r']}
      Decode: {result.get('latency_analysis', {}).get('p50_ms', 'N/A')} ms/tok (P50)
      Growth: {result.get('latency_analysis', {}).get('latency_growth_pct', 'N/A')}% over sequence

    {C['m']}Eviction (50% budget):{C['r']}
      Keeps {eviction_results[2]['h2o_attention_coverage']:.1%} of attention mass
      Saves {eviction_results[2]['memory_saved_mb']:.1f} MB ({eviction_results[2]['compression_ratio']:.1f}× compression)

    {C['v']}Output preview:{C['r']} {output_text[:150]}...
    """)

    # Clean up
    del past_key_values, outputs, next_logits, next_token, input_ids_gpu
    import torch
    torch.cuda.empty_cache()

    return result


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KV Cache Deep Profiler for LLM Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 kv_cache_profiler.py --model Qwen/Qwen2.5-7B-Instruct
  python3 kv_cache_profiler.py --model Qwen/Qwen2.5-7B-Instruct --max-tokens 512
  python3 kv_cache_profiler.py --model Qwen/Qwen2.5-7B-Instruct --prompt "your prompt"
        """,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt (default: built-in KV cache question)")
    parser.add_argument("--output-dir", default="./results/kv_cache")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--hf-token", default=None)

    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print(f"{C['e']}CUDA not available. This profiler requires a GPU.{C['r']}")
        sys.exit(1)

    banner("KV CACHE DEEP PROFILER", "▓")
    kv("Model", args.model)
    kv("Max tokens", str(args.max_tokens))
    kv("Dtype", args.dtype)
    kv("Device", str(device))
    kv("GPU", torch.cuda.get_device_name(0))
    free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
    total_gb = torch.cuda.mem_get_info()[1] / (1024**3)
    kv("GPU Memory", f"{free_gb:.1f} / {total_gb:.1f} GB free")

    # ── Load Model ────────────────────────────────────────────
    section("Loading Model")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",  # Need eager attention to get attention weights
    ).to(device)
    model.eval()

    t1 = time.perf_counter()
    kv("Load time", f"{t1 - t0:.1f}", "s")

    model_info = get_model_kv_config(model)
    n_params = sum(p.numel() for p in model.parameters())
    kv("Parameters", f"{n_params / 1e9:.2f}", "B")
    kv("Config", f"{model_info['num_layers']}L / "
       f"{model_info['num_attention_heads']}Q / "
       f"{model_info['num_kv_heads']}KV / "
       f"dim {model_info['head_dim']}")
    kv("GPU after load", f"{gpu_mem_mb():.0f}", "MB")

    # ── Warmup ────────────────────────────────────────────────
    section("Warmup (1 short generation)")
    with torch.no_grad():
        warm_ids = tokenizer.encode("Hello", return_tensors="pt").to(device)
        warm_out = model.generate(warm_ids, max_new_tokens=8, do_sample=False)
        del warm_ids, warm_out
    torch.cuda.empty_cache()
    print(f"    {C['ok']}Warmup complete{C['r']}")

    # ── Profile ───────────────────────────────────────────────
    prompt = args.prompt or DEFAULT_PROMPT
    result = profile_kv_cache(model, tokenizer, prompt, args.max_tokens, device)

    # ── Save Results ──────────────────────────────────────────
    model_safe = args.model.replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{model_safe}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / "kv_cache_results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    section("Results Saved")
    kv("Output", str(result_path))

    # ── Print decode latency curve (ASCII) ────────────────────
    latencies = [t["decode_latency_ms"] for t in result["per_token"]
                 if "decode_latency_ms" in t]
    if len(latencies) >= 8:
        section("Decode Latency Curve (per-token)")
        _ascii_chart(latencies, width=60, height=12, xlabel="Token Position",
                     ylabel="Latency (ms)")

    # ── Print attention heatmap (ASCII) ───────────────────────
    importance = result["attention_data"]["importance_scores"]
    if len(importance) >= 8:
        section("Token Importance (H2O-style cumulative attention)")
        _ascii_chart(importance[:min(len(importance), 80)], width=60, height=8,
                     xlabel="Token Position", ylabel="Importance")

    banner("PROFILING COMPLETE", "▓")


def _ascii_chart(values, width=60, height=10, xlabel="", ylabel=""):
    """Minimal ASCII bar chart for terminal visualization."""
    n = len(values)
    v_min = min(values)
    v_max = max(values)
    v_range = v_max - v_min if v_max != v_min else 1

    # Bucket values into width bins
    bins = [[] for _ in range(width)]
    for i, v in enumerate(values):
        bin_idx = min(int(i / n * width), width - 1)
        bins[bin_idx].append(v)

    bin_avgs = [sum(b)/len(b) if b else 0 for b in bins]

    print(f"    {C['d']}{ylabel}{C['r']}")
    for row in range(height - 1, -1, -1):
        threshold = v_min + (row / (height - 1)) * v_range
        line = ""
        for b_avg in bin_avgs:
            if b_avg >= threshold:
                line += "█"
            elif b_avg >= threshold - v_range / height * 0.5:
                line += "▄"
            else:
                line += " "
        label = f"{threshold:>8.1f}" if row == height - 1 or row == 0 else " " * 8
        print(f"    {C['d']}{label}{C['r']} │{C['m']}{line}{C['r']}│")

    print(f"    {' ' * 8} └{'─' * width}┘")
    print(f"    {' ' * 8}  {xlabel}")
    print(f"    {C['d']}  n={n}  min={v_min:.2f}  max={v_max:.2f}  "
          f"mean={sum(values)/len(values):.2f}{C['r']}")


if __name__ == "__main__":
    main()
