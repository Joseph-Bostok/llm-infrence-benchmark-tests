#!/usr/bin/env python3
"""
PInsight Parameter Space Catalog

Single source of truth for every tunable parameter in the LLM inference
optimization stack. Organized by category and algorithm.

Availability tiers
──────────────────
  native    Stock vLLM CLI flag — usable today on Aries, no patches needed
  model     Requires loading a pre-quantized model checkpoint (AWQ/GPTQ/FP8)
  research  Requires custom vLLM patch or standalone implementation

Usage
─────
    python3 parameter_space.py                       # Full table, all params
    python3 parameter_space.py --native              # Native vLLM params only
    python3 parameter_space.py --tier research       # Research-only params
    python3 parameter_space.py --category TOKEN_EVICTION
    python3 parameter_space.py --algorithm SnapKV
    python3 parameter_space.py --algorithms          # List all algorithms
    python3 parameter_space.py --counts              # Stats per category/tier
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


# ─── Types ───────────────────────────────────────────────────────────────────

class Tier(str, Enum):
    NATIVE   = "native"    # Stock vLLM flag, usable today
    MODEL    = "model"     # Pre-quantized model variant required
    RESEARCH = "research"  # Custom implementation / patched vLLM


class PType(str, Enum):
    FLOAT       = "float"
    INT         = "int"
    BOOL        = "bool"
    CATEGORICAL = "cat"


@dataclass
class Param:
    name: str
    description: str
    ptype: PType
    default: Any
    values: List[Any]      # Discrete values to sweep over
    category: str          # SYSTEM | WEIGHT_QUANT | TOKEN_EVICTION | KV_QUANT | LOAD_PROFILE
    algorithm: str         # Owning technique
    tier: Tier
    vllm_flag: Optional[str] = None   # e.g. "--gpu-memory-utilization"
    notes: Optional[str] = None


# ─── Parameter Space ─────────────────────────────────────────────────────────

PARAMETER_SPACE: List[Param] = [

    # ═══════════════════════════════════════════════════════════════════════
    # SYSTEM — Native vLLM parameters (all usable today with stock vLLM)
    # ═══════════════════════════════════════════════════════════════════════

    # vLLM Core ──────────────────────────────────────────────────────────────
    Param("gpu_memory_utilization",
          "Fraction of GPU memory for KV cache + weights",
          PType.FLOAT, 0.90,
          [0.75, 0.80, 0.85, 0.90, 0.92, 0.95],
          "SYSTEM", "vLLM Core", Tier.NATIVE,
          vllm_flag="--gpu-memory-utilization"),

    Param("block_size",
          "PagedAttention KV cache block size in tokens",
          PType.INT, 16,
          [8, 16, 32],
          "SYSTEM", "vLLM Core", Tier.NATIVE,
          vllm_flag="--block-size",
          notes="Power-of-2 only. Larger = less fragmentation, more waste per partial block"),

    Param("max_model_len",
          "Max sequence length (prompt + output tokens)",
          PType.INT, None,
          [4096, 8192, 16384, 32768, 65536],
          "SYSTEM", "vLLM Core", Tier.NATIVE,
          vllm_flag="--max-model-len",
          notes="Determines peak KV cache VRAM. Lower = faster, less memory"),

    Param("max_num_seqs",
          "Max concurrent sequences per scheduler iteration",
          PType.INT, 256,
          [32, 64, 128, 256, 512],
          "SYSTEM", "vLLM Core", Tier.NATIVE,
          vllm_flag="--max-num-seqs"),

    Param("enforce_eager",
          "Disable CUDA graphs (use eager execution — needed for some profilers)",
          PType.BOOL, False,
          [False, True],
          "SYSTEM", "vLLM Core", Tier.NATIVE,
          vllm_flag="--enforce-eager",
          notes="Set True when profiling with Kineto/Nsys to avoid graph-capture overhead"),

    Param("dtype",
          "Weight and activation precision",
          PType.CATEGORICAL, "auto",
          ["auto", "float16", "bfloat16"],
          "SYSTEM", "vLLM Core", Tier.NATIVE,
          vllm_flag="--dtype"),

    # vLLM Prefill ───────────────────────────────────────────────────────────
    Param("enable_chunked_prefill",
          "Interleave prefill chunks with decode steps (reduces TTFT variance)",
          PType.BOOL, False,
          [False, True],
          "SYSTEM", "vLLM Prefill", Tier.NATIVE,
          vllm_flag="--enable-chunked-prefill"),

    Param("max_num_batched_tokens",
          "Token budget per scheduler step (controls chunked prefill chunk size)",
          PType.INT, 8192,
          [2048, 4096, 8192, 16384, 32768],
          "SYSTEM", "vLLM Prefill", Tier.NATIVE,
          vllm_flag="--max-num-batched-tokens",
          notes="Higher = better GPU utilization during prefill"),

    Param("num_scheduler_steps",
          "Forward passes per scheduler iteration (multi-step scheduling)",
          PType.INT, 1,
          [1, 2, 4, 8],
          "SYSTEM", "vLLM Prefill", Tier.NATIVE,
          vllm_flag="--num-scheduler-steps",
          notes="Higher = better throughput, higher per-step latency. Requires vLLM ≥0.4"),

    # Prefix Caching ─────────────────────────────────────────────────────────
    Param("enable_prefix_caching",
          "Reuse KV cache for shared prompt prefixes across requests",
          PType.BOOL, False,
          [False, True],
          "SYSTEM", "Prefix Caching", Tier.NATIVE,
          vllm_flag="--enable-prefix-caching",
          notes="High-value for RAG and chat workloads with shared system prompts"),

    # KV Cache Dtype ─────────────────────────────────────────────────────────
    Param("kv_cache_dtype",
          "Stored KV cache tensor precision",
          PType.CATEGORICAL, "auto",
          ["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
          "SYSTEM", "KV Cache Dtype", Tier.NATIVE,
          vllm_flag="--kv-cache-dtype",
          notes="fp8 halves KV cache memory vs bf16/fp16 with small accuracy impact"),

    # Memory Offload ─────────────────────────────────────────────────────────
    Param("swap_space",
          "CPU RAM for KV cache block swapping (GB) — legacy flag",
          PType.INT, 4,
          [0, 4, 8, 16],
          "SYSTEM", "Memory Offload", Tier.NATIVE,
          vllm_flag="--swap-space",
          notes="Deprecated in vLLM ≥0.5; prefer --cpu-offload-gb"),

    Param("cpu_offload_gb",
          "CPU memory for KV cache offloading (GB) — replaces swap_space",
          PType.FLOAT, 0,
          [0, 4, 8, 16, 32],
          "SYSTEM", "Memory Offload", Tier.NATIVE,
          vllm_flag="--cpu-offload-gb"),

    # Parallelism ────────────────────────────────────────────────────────────
    Param("tensor_parallel_size",
          "Number of GPUs for tensor parallelism (model sharding across GPUs)",
          PType.INT, 1,
          [1, 2, 4, 8],
          "SYSTEM", "Parallelism", Tier.NATIVE,
          vllm_flag="--tensor-parallel-size",
          notes="Required for models >40 GB on a single GPU"),

    Param("pipeline_parallel_size",
          "Number of pipeline stages (layer sharding across GPUs)",
          PType.INT, 1,
          [1, 2, 4],
          "SYSTEM", "Parallelism", Tier.NATIVE,
          vllm_flag="--pipeline-parallel-size"),

    # ═══════════════════════════════════════════════════════════════════════
    # WEIGHT_QUANT — Requires a pre-quantized model checkpoint
    # ═══════════════════════════════════════════════════════════════════════

    Param("quantization",
          "Weight quantization method applied at model load time",
          PType.CATEGORICAL, None,
          ["awq", "gptq", "fp8", "bitsandbytes", "compressed-tensors", "marlin"],
          "WEIGHT_QUANT", "Weight Quantization", Tier.MODEL,
          vllm_flag="--quantization",
          notes="Requires pre-quantized checkpoint. AWQ/GPTQ are the most available"),

    Param("awq_group_size",
          "AWQ quantization group size (per-group scales for weight matrices)",
          PType.INT, 128,
          [32, 64, 128],
          "WEIGHT_QUANT", "AWQ", Tier.MODEL,
          notes="Smaller = better accuracy, larger = faster dequant"),

    Param("gptq_bits",
          "GPTQ weight precision",
          PType.INT, 4,
          [2, 3, 4, 8],
          "WEIGHT_QUANT", "GPTQ", Tier.MODEL,
          notes="4-bit standard. Lower = smaller model, more accuracy loss"),

    Param("bnb_load_in_4bit",
          "bitsandbytes NF4 quantization (load model in 4-bit)",
          PType.BOOL, False,
          [False, True],
          "WEIGHT_QUANT", "bitsandbytes", Tier.MODEL,
          notes="Easier to use than AWQ/GPTQ but slower for production serving"),

    # ═══════════════════════════════════════════════════════════════════════
    # TOKEN_EVICTION — Research algorithms (require custom vLLM patch)
    # ═══════════════════════════════════════════════════════════════════════

    # H2O ────────────────────────────────────────────────────────────────────
    Param("h2o_kv_cache_budget",
          "Fraction of tokens to retain by cumulative attention score",
          PType.FLOAT, 1.0,
          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0],
          "TOKEN_EVICTION", "H2O", Tier.RESEARCH,
          notes="0.5 = keep top-50% of tokens by H2O score + recent window"),

    Param("h2o_heavy_hitter_fraction",
          "Share of budget for heavy-hitter tokens (remainder = recent window)",
          PType.FLOAT, 0.9,
          [0.7, 0.8, 0.9, 1.0],
          "TOKEN_EVICTION", "H2O", Tier.RESEARCH),

    Param("h2o_recent_window",
          "Number of most-recent tokens always retained (not scored)",
          PType.INT, 20,
          [0, 10, 20, 50, 100],
          "TOKEN_EVICTION", "H2O", Tier.RESEARCH),

    # SnapKV ─────────────────────────────────────────────────────────────────
    Param("snapkv_kv_cache_budget",
          "Fraction of tokens retained per attention head (per-head selection)",
          PType.FLOAT, 1.0,
          [0.2, 0.3, 0.5, 0.6, 0.75, 1.0],
          "TOKEN_EVICTION", "SnapKV", Tier.RESEARCH,
          notes="Per-head selection catches head-specific patterns H2O misses"),

    Param("snapkv_observation_window",
          "Last N prompt tokens used to score importance (observation window)",
          PType.INT, 16,
          [8, 16, 32, 64, 128],
          "TOKEN_EVICTION", "SnapKV", Tier.RESEARCH,
          notes="Larger = more accurate importance estimate but more prefill compute"),

    Param("snapkv_kernel_size",
          "Pooling kernel size for smoothing per-head attention scores",
          PType.INT, 5,
          [3, 5, 7],
          "TOKEN_EVICTION", "SnapKV", Tier.RESEARCH),

    # StreamingLLM ────────────────────────────────────────────────────────────
    Param("streaming_sink_tokens",
          "Initial 'attention sink' tokens always retained",
          PType.INT, 4,
          [1, 2, 4, 8, 16],
          "TOKEN_EVICTION", "StreamingLLM", Tier.RESEARCH,
          notes="Softmax forces attention mass onto early tokens — dropping them causes catastrophic failure"),

    Param("streaming_window_size",
          "Number of recent tokens in the sliding window",
          PType.INT, 256,
          [64, 128, 256, 512, 1024],
          "TOKEN_EVICTION", "StreamingLLM", Tier.RESEARCH,
          notes="Total KV size = sinks + window. Enables theoretically infinite context"),

    # ChunkKV ────────────────────────────────────────────────────────────────
    Param("chunkkv_kv_cache_budget",
          "Fraction of KV chunks retained (preserves linguistic coherence)",
          PType.FLOAT, 1.0,
          [0.3, 0.4, 0.5, 0.6, 0.75, 1.0],
          "TOKEN_EVICTION", "ChunkKV", Tier.RESEARCH,
          notes="Outperforms H2O/SnapKV on LongBench at same compression by keeping context contiguous"),

    Param("chunkkv_chunk_size",
          "Number of tokens per eviction chunk (contiguous group)",
          PType.INT, 16,
          [4, 8, 16, 32, 64],
          "TOKEN_EVICTION", "ChunkKV", Tier.RESEARCH,
          notes="Larger = better coherence, less granularity"),

    Param("chunkkv_reuse_layer_indices",
          "Reuse chunk selection from adjacent layers to cut overhead",
          PType.BOOL, True,
          [True, False],
          "TOKEN_EVICTION", "ChunkKV", Tier.RESEARCH),

    # CurDKV ─────────────────────────────────────────────────────────────────
    Param("curdkv_kv_cache_budget",
          "Fraction of tokens retained by CUR leverage-score ranking",
          PType.FLOAT, 1.0,
          [0.3, 0.5, 0.7, 1.0],
          "TOKEN_EVICTION", "CurDKV", Tier.RESEARCH,
          notes="Ranks by actual contribution to attention output (Q·K^T · V), not just attention score"),

    Param("curdkv_leverage_sampling",
          "Sample proportional to leverage scores (True) vs hard top-k (False)",
          PType.BOOL, False,
          [False, True],
          "TOKEN_EVICTION", "CurDKV", Tier.RESEARCH),

    # R-KV ────────────────────────────────────────────────────────────────────
    Param("rkv_kv_cache_budget",
          "Fraction of tokens retained after redundancy-aware eviction",
          PType.FLOAT, 1.0,
          [0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
          "TOKEN_EVICTION", "R-KV", Tier.RESEARCH,
          notes="Designed for CoT/reasoning models. Penalizes near-duplicate tokens in chain-of-thought"),

    Param("rkv_redundancy_penalty",
          "Redundancy penalty weight in importance score (0 = pure attn, 1 = pure diversity)",
          PType.FLOAT, 0.5,
          [0.0, 0.25, 0.5, 0.75, 1.0],
          "TOKEN_EVICTION", "R-KV", Tier.RESEARCH),

    Param("rkv_similarity_metric",
          "Metric for computing inter-token redundancy",
          PType.CATEGORICAL, "cosine",
          ["cosine", "l2", "dot_product"],
          "TOKEN_EVICTION", "R-KV", Tier.RESEARCH),

    # ═══════════════════════════════════════════════════════════════════════
    # KV_QUANT — KV cache quantization (research, requires custom attention)
    # ═══════════════════════════════════════════════════════════════════════

    # KIVI ────────────────────────────────────────────────────────────────────
    Param("kivi_bits",
          "KV cache quantization precision",
          PType.INT, 16,
          [2, 4, 8, 16],
          "KV_QUANT", "KIVI", Tier.RESEARCH,
          notes="2-bit = 2.6× memory reduction. Keys: per-channel quant, Values: per-token quant"),

    Param("kivi_residual_length",
          "Most-recent KV pairs kept at full precision (not quantized)",
          PType.INT, 128,
          [32, 64, 128, 256],
          "KV_QUANT", "KIVI", Tier.RESEARCH,
          notes="Recent tokens kept at fp16; older tokens quantized"),

    Param("kivi_group_size",
          "Sub-channel quantization group size (smaller = better precision)",
          PType.INT, 32,
          [8, 16, 32, 64],
          "KV_QUANT", "KIVI", Tier.RESEARCH),

    # MiniKV ──────────────────────────────────────────────────────────────────
    Param("minikv_bits",
          "KV cache quantization bits (always combined with token eviction)",
          PType.INT, 2,
          [2, 4],
          "KV_QUANT", "MiniKV", Tier.RESEARCH,
          notes="Key insight: more tokens at INT2 beats fewer tokens at FP16"),

    Param("minikv_token_budget",
          "Fraction of tokens retained after eviction (then quantized)",
          PType.FLOAT, 0.5,
          [0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
          "KV_QUANT", "MiniKV", Tier.RESEARCH),

    Param("minikv_layer_strategy",
          "Per-layer budget allocation strategy",
          PType.CATEGORICAL, "discriminative",
          ["uniform", "discriminative", "entropy"],
          "KV_QUANT", "MiniKV", Tier.RESEARCH,
          notes="discriminative = different budgets per layer based on sensitivity analysis"),

    # ShadowKV ────────────────────────────────────────────────────────────────
    Param("shadowkv_rank",
          "Low-rank approximation rank for key compression",
          PType.INT, 96,
          [32, 64, 96, 128, 192],
          "KV_QUANT", "ShadowKV", Tier.RESEARCH,
          notes="Pre-RoPE keys are low-rank. Keys stay on GPU; full values offloaded to CPU"),

    Param("shadowkv_chunk_size",
          "Chunk size for sparse KV reconstruction during decode",
          PType.INT, 8,
          [4, 8, 16, 32],
          "KV_QUANT", "ShadowKV", Tier.RESEARCH),

    # ═══════════════════════════════════════════════════════════════════════
    # LOAD_PROFILE — Evaluation / load testing parameters
    # ═══════════════════════════════════════════════════════════════════════

    Param("request_rate",
          "Request arrival rate for load testing (requests/second)",
          PType.FLOAT, 1.0,
          [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
          "LOAD_PROFILE", "GuideLLM", Tier.NATIVE,
          notes="Use 'sweep' mode in GuideLLM to auto-find the saturation point"),

    Param("rate_type",
          "Arrival pattern for load testing",
          PType.CATEGORICAL, "constant",
          ["constant", "poisson", "sweep", "synchronous"],
          "LOAD_PROFILE", "GuideLLM", Tier.NATIVE,
          notes="poisson = realistic burst pattern; sweep = throughput-vs-latency curve"),

    Param("max_output_tokens",
          "Maximum tokens to generate per request",
          PType.INT, 128,
          [64, 128, 256, 512, 1024, 2048],
          "LOAD_PROFILE", "GuideLLM", Tier.NATIVE),

    Param("concurrency",
          "Number of concurrent client workers sending requests",
          PType.INT, 1,
          [1, 2, 4, 8, 16, 32],
          "LOAD_PROFILE", "GuideLLM", Tier.NATIVE),

    Param("ttft_deadline_ms",
          "SLO deadline for Time-To-First-Token in ms (Etalon)",
          PType.FLOAT, 500.0,
          [100, 200, 500, 1000, 2000],
          "LOAD_PROFILE", "Etalon", Tier.NATIVE,
          notes="Requests exceeding this count as SLO violations"),

    Param("tbt_deadline_ms",
          "SLO deadline for Time-Between-Tokens in ms (Etalon)",
          PType.FLOAT, 50.0,
          [20, 50, 100, 200],
          "LOAD_PROFILE", "Etalon", Tier.NATIVE),
]


# ─── Query helpers ───────────────────────────────────────────────────────────

def get_by_category(cat: str) -> List[Param]:
    return [p for p in PARAMETER_SPACE if p.category == cat]


def get_by_algorithm(algo: str) -> List[Param]:
    return [p for p in PARAMETER_SPACE if p.algorithm.lower() == algo.lower()]


def get_by_tier(tier: Tier) -> List[Param]:
    return [p for p in PARAMETER_SPACE if p.tier == tier]


def list_algorithms() -> Dict[str, List[str]]:
    """Returns {category: [algorithm, ...]} ordered dict."""
    result: Dict[str, List[str]] = {}
    for p in PARAMETER_SPACE:
        result.setdefault(p.category, [])
        if p.algorithm not in result[p.category]:
            result[p.category].append(p.algorithm)
    return result


def get_native_sweep_values() -> Dict[str, List[Any]]:
    """Return {param_name: values} for all native vLLM params (for use in sweeps)."""
    return {p.name: p.values for p in PARAMETER_SPACE
            if p.tier == Tier.NATIVE and p.vllm_flag}


# ─── Table printer ───────────────────────────────────────────────────────────

_TIER_COLOR = {
    Tier.NATIVE:   "\033[92m",  # green
    Tier.MODEL:    "\033[93m",  # yellow
    Tier.RESEARCH: "\033[94m",  # blue
}
_RESET = "\033[0m"


def _fmt_values(values: List[Any], max_len: int = 48) -> str:
    s = ", ".join(str(v) for v in values)
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    return s


def _fmt_default(d: Any) -> str:
    if d is None:
        return "—"
    return str(d)


def print_table(
    params: Optional[List[Param]] = None,
    use_color: bool = True,
    show_notes: bool = False,
) -> None:
    """Print a formatted parameter space table."""
    rows = params if params is not None else PARAMETER_SPACE

    W_NAME    = 34
    W_ALGO    = 20
    W_TYPE    = 5
    W_DEFAULT = 7
    W_TIER    = 8
    W_VALUES  = 46
    total = W_NAME + W_ALGO + W_TYPE + W_DEFAULT + W_TIER + W_VALUES + 12

    def _hr(char="─"):
        print(char * total)

    def _header():
        print(
            f"{'PARAMETER':<{W_NAME}}  "
            f"{'ALGORITHM':<{W_ALGO}}  "
            f"{'TYPE':<{W_TYPE}}  "
            f"{'DEFLT':<{W_DEFAULT}}  "
            f"{'TIER':<{W_TIER}}  "
            f"{'SWEEP VALUES'}"
        )

    print()
    _hr("═")
    print("PINSIGHT PARAMETER SPACE CATALOG")
    _hr("═")

    current_cat = None
    for p in rows:
        if p.category != current_cat:
            current_cat = p.category
            print()
            _hr()
            label = {
                "SYSTEM":          "SYSTEM  —  Native vLLM parameters (usable today on Aries)",
                "WEIGHT_QUANT":    "WEIGHT_QUANT  —  Weight quantization (requires pre-quantized model)",
                "TOKEN_EVICTION":  "TOKEN_EVICTION  —  KV cache eviction algorithms (research / custom patch)",
                "KV_QUANT":        "KV_QUANT  —  KV cache quantization (research / custom attention backend)",
                "LOAD_PROFILE":    "LOAD_PROFILE  —  Evaluation / load testing parameters",
            }.get(current_cat, current_cat)
            print(f"  {label}")
            _hr()
            _header()
            _hr()

        tier_str = p.tier.value
        if use_color:
            color = _TIER_COLOR.get(p.tier, "")
            tier_str = f"{color}{tier_str}{_RESET}"

        print(
            f"  {p.name:<{W_NAME}}"
            f"  {p.algorithm:<{W_ALGO}}"
            f"  {p.ptype.value:<{W_TYPE}}"
            f"  {_fmt_default(p.default):<{W_DEFAULT}}"
            f"  {tier_str:<{W_TIER + (len(tier_str) - len(p.tier.value)) * use_color}}"
            f"  {_fmt_values(p.values, W_VALUES)}"
        )
        if show_notes and p.notes:
            print(f"    {'':>{W_NAME + 2}}↳ {p.notes}")

    print()
    _hr("═")

    # Summary stats
    native   = sum(1 for p in rows if p.tier == Tier.NATIVE)
    model    = sum(1 for p in rows if p.tier == Tier.MODEL)
    research = sum(1 for p in rows if p.tier == Tier.RESEARCH)
    total_params = len(rows)
    print(
        f"  {total_params} parameters  |  "
        f"native: {native}  |  model: {model}  |  research: {research}"
    )
    _hr("═")
    print()


def print_algorithms() -> None:
    """Print all algorithms organized by category."""
    algos = list_algorithms()
    categories = ["SYSTEM", "WEIGHT_QUANT", "TOKEN_EVICTION", "KV_QUANT", "LOAD_PROFILE"]

    print()
    print("═" * 60)
    print("ALGORITHMS BY CATEGORY")
    print("═" * 60)

    for cat in categories:
        if cat not in algos:
            continue
        tier_example = next(
            (p.tier.value for p in PARAMETER_SPACE if p.category == cat), "")
        print(f"\n  {cat}  [{tier_example}]")
        print("  " + "─" * 40)
        for algo in algos[cat]:
            params = get_by_algorithm(algo)
            param_names = ", ".join(p.name for p in params)
            print(f"    {algo:<24}  {len(params)} params  →  {param_names[:60]}")

    print()


def print_counts() -> None:
    """Print parameter counts broken down by category and tier."""
    from collections import Counter

    by_cat_tier: Counter = Counter()
    for p in PARAMETER_SPACE:
        by_cat_tier[(p.category, p.tier.value)] += 1

    categories = ["SYSTEM", "WEIGHT_QUANT", "TOKEN_EVICTION", "KV_QUANT", "LOAD_PROFILE"]
    tiers = ["native", "model", "research"]

    W = 16
    print()
    print(f"  {'CATEGORY':<20}  {'native':>{W}}  {'model':>{W}}  {'research':>{W}}  {'total':>{W}}")
    print("  " + "─" * (20 + 3 * (W + 2) + W + 8))
    for cat in categories:
        counts = {t: by_cat_tier[(cat, t)] for t in tiers}
        total = sum(counts.values())
        if total == 0:
            continue
        print(
            f"  {cat:<20}"
            f"  {counts['native']:>{W}}"
            f"  {counts['model']:>{W}}"
            f"  {counts['research']:>{W}}"
            f"  {total:>{W}}"
        )
    print("  " + "─" * (20 + 3 * (W + 2) + W + 8))
    grand = len(PARAMETER_SPACE)
    n = sum(1 for p in PARAMETER_SPACE if p.tier == Tier.NATIVE)
    m = sum(1 for p in PARAMETER_SPACE if p.tier == Tier.MODEL)
    r = sum(1 for p in PARAMETER_SPACE if p.tier == Tier.RESEARCH)
    print(f"  {'TOTAL':<20}  {n:>{W}}  {m:>{W}}  {r:>{W}}  {grand:>{W}}")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PInsight Parameter Space Catalog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--native", action="store_true",
                        help="Show native vLLM params only")
    parser.add_argument("--tier", choices=["native", "model", "research"],
                        help="Filter by availability tier")
    parser.add_argument("--category", type=str,
                        help="Filter by category (SYSTEM, WEIGHT_QUANT, TOKEN_EVICTION, KV_QUANT, LOAD_PROFILE)")
    parser.add_argument("--algorithm", type=str,
                        help="Filter by algorithm (e.g. H2O, SnapKV, KIVI)")
    parser.add_argument("--algorithms", action="store_true",
                        help="List all algorithms by category")
    parser.add_argument("--counts", action="store_true",
                        help="Show parameter counts by category and tier")
    parser.add_argument("--notes", action="store_true",
                        help="Include notes in the table")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable terminal color")
    args = parser.parse_args()

    if args.algorithms:
        print_algorithms()
        return

    if args.counts:
        print_counts()
        return

    params = PARAMETER_SPACE

    if args.native:
        params = [p for p in params if p.tier == Tier.NATIVE]
    elif args.tier:
        params = [p for p in params if p.tier.value == args.tier]

    if args.category:
        cat = args.category.upper()
        params = [p for p in params if p.category == cat]

    if args.algorithm:
        params = get_by_algorithm(args.algorithm)

    print_table(params, use_color=not args.no_color, show_notes=args.notes)


if __name__ == "__main__":
    main()
