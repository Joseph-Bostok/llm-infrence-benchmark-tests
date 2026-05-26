#!/usr/bin/env python3
"""
PInsight Multi-Level Tuning Engine

Takes workload classification + profiling data and produces optimal
vLLM configuration parameters across three levels:

    Token Level  - KV pruning budget, heavy-hitter threshold, sink tokens
    Model Level  - Chunk size, layer-wise budgets
    System Level - vLLM memory utilization, block size, prefix caching, swap

This is the core decision engine in PInsight's in-situ loop:
    Profile -> Classify -> [Tune] -> Monitor -> Repeat
"""

import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from workload_classifier import WorkloadType, WorkloadClassification


@dataclass
class TokenLevelConfig:
    """Token-level KV cache tuning parameters."""
    kv_cache_budget: float = 1.0          # Fraction of tokens to retain (0.0-1.0)
    heavy_hitter_threshold: float = 0.01  # Attention score cutoff for heavy hitters
    sink_token_count: int = 4             # Initial tokens always retained
    sliding_window_size: int = 256        # Recent tokens always retained
    eviction_policy: str = "h2o"          # "h2o", "snapkv", "streaming"


@dataclass
class ModelLevelConfig:
    """Model-level KV cache tuning parameters."""
    chunk_size: int = 16                  # Tokens per chunk for chunk-level eviction
    use_chunked_eviction: bool = False    # Chunk-level vs token-level
    layer_budget_strategy: str = "uniform"  # "uniform", "entropy", "discriminative"
    layer_budgets: Optional[Dict[int, float]] = None  # Per-layer retention ratios


@dataclass
class SystemLevelConfig:
    """System-level vLLM configuration parameters."""
    gpu_memory_utilization: float = 0.90  # Fraction of GPU memory for KV cache
    max_num_batched_tokens: int = 8192    # Chunked prefill token limit
    block_size: int = 16                  # PagedAttention block size
    enable_prefix_caching: bool = False   # Automatic prefix caching
    swap_space_gb: float = 4.0            # CPU swap space for KV cache offload
    max_model_len: Optional[int] = None   # Max sequence length
    enforce_eager: bool = False           # Disable CUDA graphs (for profiling)


@dataclass
class TuningConfig:
    """Complete multi-level tuning configuration."""
    workload_type: str
    confidence: float
    token_level: TokenLevelConfig
    model_level: ModelLevelConfig
    system_level: SystemLevelConfig
    reasoning: List[str]  # Why these parameters were chosen

    def to_vllm_args(self) -> Dict:
        """Convert to vLLM server launch arguments."""
        args = {
            "gpu_memory_utilization": self.system_level.gpu_memory_utilization,
            "max_num_batched_tokens": self.system_level.max_num_batched_tokens,
            "block_size": self.system_level.block_size,
            "enable_prefix_caching": self.system_level.enable_prefix_caching,
            "enforce_eager": self.system_level.enforce_eager,
        }
        
        # Check if vllm supports --swap-space
        import subprocess
        supports_swap = True
        try:
            res = subprocess.run(
                ["python3", "-m", "vllm.entrypoints.openai.api_server", "--help"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
            )
            supports_swap = "--swap-space" in res.stdout
        except Exception:
            pass
            
        if supports_swap:
            args["swap_space"] = int(self.system_level.swap_space_gb)
            
        if self.system_level.max_model_len:
            args["max_model_len"] = self.system_level.max_model_len
        return args

    def to_vllm_cli(self, model: str, port: int = 8000) -> str:
        """Generate vLLM server launch command."""
        args = self.to_vllm_args()
        parts = [
            f"python -m vllm.entrypoints.openai.api_server",
            f"--model {model}",
            f"--port {port}",
        ]
        for k, v in args.items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    parts.append(flag)
            else:
                parts.append(f"{flag} {v}")
        return " \\\n    ".join(parts)


# --- Tuning Strategies per Workload ---

def _tune_dialogue(confidence: float) -> TuningConfig:
    """Dialogue: attention sinks + sliding window, moderate pruning."""
    reasons = [
        "Dialogue workloads concentrate attention on initial tokens (sinks) "
        "and the most recent turns.",
        "Aggressive pruning of middle-conversation turns is safe because "
        "the model rarely attends back to them.",
        "Prefix caching helps when system prompts are shared across users.",
        "Moderate memory utilization — dialogue tends to have many concurrent "
        "sessions with short-to-medium contexts.",
    ]
    return TuningConfig(
        workload_type="dialogue",
        confidence=confidence,
        token_level=TokenLevelConfig(
            kv_cache_budget=0.50,
            heavy_hitter_threshold=0.01,
            sink_token_count=4,
            sliding_window_size=512,
            eviction_policy="streaming",
        ),
        model_level=ModelLevelConfig(
            chunk_size=16,
            use_chunked_eviction=False,
            layer_budget_strategy="uniform",
        ),
        system_level=SystemLevelConfig(
            gpu_memory_utilization=0.85,
            max_num_batched_tokens=4096,
            block_size=16,
            enable_prefix_caching=True,
            swap_space_gb=4.0,
            max_model_len=8192,
        ),
        reasoning=reasons,
    )


def _tune_rag(confidence: float) -> TuningConfig:
    """RAG: preserve document passages, chunk-level eviction."""
    reasons = [
        "RAG workloads inject long document passages that must stay coherent.",
        "Chunk-level eviction preserves passage integrity — token-level "
        "would fragment retrieved documents.",
        "Higher cache budget (75%) because retrieval accuracy depends on "
        "having the full context available.",
        "Large max_num_batched_tokens for efficient prefill of long prompts.",
        "Prefix caching critical — many RAG queries share the same "
        "system prompt and instruction prefix.",
    ]
    return TuningConfig(
        workload_type="rag",
        confidence=confidence,
        token_level=TokenLevelConfig(
            kv_cache_budget=0.75,
            heavy_hitter_threshold=0.005,
            sink_token_count=4,
            sliding_window_size=256,
            eviction_policy="snapkv",
        ),
        model_level=ModelLevelConfig(
            chunk_size=32,
            use_chunked_eviction=True,
            layer_budget_strategy="entropy",
        ),
        system_level=SystemLevelConfig(
            gpu_memory_utilization=0.92,
            max_num_batched_tokens=16384,
            block_size=16,
            enable_prefix_caching=True,
            swap_space_gb=8.0,
            max_model_len=32768,
        ),
        reasoning=reasons,
    )


def _tune_code(confidence: float) -> TuningConfig:
    """Code: scattered important tokens, token-level eviction, high budget."""
    reasons = [
        "Code workloads have scattered long-range dependencies — variable "
        "defs and function signatures from far back must be retained.",
        "Token-level eviction preferred because important tokens (identifiers, "
        "signatures) are spread across the sequence, not contiguous.",
        "High cache budget (80%) because dropping a critical variable "
        "definition causes semantically broken output.",
        "No prefix caching — code prompts are rarely identical.",
    ]
    return TuningConfig(
        workload_type="code",
        confidence=confidence,
        token_level=TokenLevelConfig(
            kv_cache_budget=0.80,
            heavy_hitter_threshold=0.005,
            sink_token_count=8,
            sliding_window_size=512,
            eviction_policy="snapkv",
        ),
        model_level=ModelLevelConfig(
            chunk_size=8,
            use_chunked_eviction=False,
            layer_budget_strategy="discriminative",
        ),
        system_level=SystemLevelConfig(
            gpu_memory_utilization=0.90,
            max_num_batched_tokens=8192,
            block_size=16,
            enable_prefix_caching=False,
            swap_space_gb=4.0,
            max_model_len=16384,
        ),
        reasoning=reasons,
    )


def _tune_reasoning(confidence: float) -> TuningConfig:
    """Reasoning: redundant CoT tokens, aggressive pruning."""
    reasons = [
        "Reasoning/CoT workloads generate long, repetitive traces with many "
        "redundant tokens ('let me reconsider', 'going back to').",
        "Aggressive pruning is safe because redundant reasoning tokens carry "
        "duplicate information.",
        "Token-level eviction can detect and remove near-duplicate tokens.",
        "Lower memory utilization for longer generation runs.",
    ]
    return TuningConfig(
        workload_type="reasoning",
        confidence=confidence,
        token_level=TokenLevelConfig(
            kv_cache_budget=0.40,
            heavy_hitter_threshold=0.02,
            sink_token_count=4,
            sliding_window_size=256,
            eviction_policy="h2o",
        ),
        model_level=ModelLevelConfig(
            chunk_size=16,
            use_chunked_eviction=False,
            layer_budget_strategy="discriminative",
        ),
        system_level=SystemLevelConfig(
            gpu_memory_utilization=0.88,
            max_num_batched_tokens=8192,
            block_size=16,
            enable_prefix_caching=False,
            swap_space_gb=4.0,
            max_model_len=32768,
        ),
        reasoning=reasons,
    )


def _tune_default(confidence: float) -> TuningConfig:
    """Conservative defaults for mixed/unknown workloads."""
    return TuningConfig(
        workload_type="default",
        confidence=confidence,
        token_level=TokenLevelConfig(),
        model_level=ModelLevelConfig(),
        system_level=SystemLevelConfig(),
        reasoning=["Using conservative defaults for unclassified workload."],
    )


TUNING_MAP = {
    WorkloadType.DIALOGUE: _tune_dialogue,
    WorkloadType.RAG: _tune_rag,
    WorkloadType.CODE: _tune_code,
    WorkloadType.REASONING: _tune_reasoning,
    WorkloadType.MIXED: _tune_default,
    WorkloadType.UNKNOWN: _tune_default,
}


def generate_tuning_config(classification: WorkloadClassification) -> TuningConfig:
    """Generate tuning config from workload classification."""
    tuner = TUNING_MAP.get(classification.workload_type, _tune_default)
    return tuner(classification.confidence)


def print_tuning_config(config: TuningConfig, model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Pretty-print a tuning configuration."""
    print(f"\n{'='*60}")
    print(f"PINSIGHT TUNING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Workload: {config.workload_type} (confidence: {config.confidence:.2f})")

    print(f"\n{'─'*60}")
    print("TOKEN LEVEL:")
    t = config.token_level
    print(f"  Cache Budget:    {t.kv_cache_budget:.0%} of tokens retained")
    print(f"  HH Threshold:    {t.heavy_hitter_threshold}")
    print(f"  Sink Tokens:     {t.sink_token_count}")
    print(f"  Sliding Window:  {t.sliding_window_size}")
    print(f"  Eviction Policy: {t.eviction_policy}")

    print(f"\n{'─'*60}")
    print("MODEL LEVEL:")
    m = config.model_level
    print(f"  Chunk Size:      {m.chunk_size} tokens")
    print(f"  Chunked Eviction:{m.use_chunked_eviction}")
    print(f"  Budget Strategy: {m.layer_budget_strategy}")

    print(f"\n{'─'*60}")
    print("SYSTEM LEVEL (vLLM):")
    s = config.system_level
    print(f"  GPU Memory Util: {s.gpu_memory_utilization:.0%}")
    print(f"  Batched Tokens:  {s.max_num_batched_tokens}")
    print(f"  Block Size:      {s.block_size}")
    print(f"  Prefix Caching:  {s.enable_prefix_caching}")
    print(f"  Swap Space:      {s.swap_space_gb} GB")
    if s.max_model_len:
        print(f"  Max Model Len:   {s.max_model_len}")

    print(f"\n{'─'*60}")
    print("REASONING:")
    for r in config.reasoning:
        print(f"  • {r}")

    print(f"\n{'─'*60}")
    print("vLLM LAUNCH COMMAND:")
    print(f"  {config.to_vllm_cli(model)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="PInsight Multi-Level Tuning Engine")
    parser.add_argument("--workload", type=str, default=None,
                        choices=["dialogue", "rag", "code", "reasoning"],
                        help="Workload type to tune for")
    parser.add_argument("--prompt", type=str, help="Classify prompt then tune")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name for vLLM command")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--compare-all", action="store_true",
                        help="Show configs for all workload types")
    args = parser.parse_args()

    if args.compare_all:
        print("\nComparing tuning configs for all workload types:\n")
        configs = {}
        for wt in [WorkloadType.DIALOGUE, WorkloadType.RAG,
                    WorkloadType.CODE, WorkloadType.REASONING]:
            tuner = TUNING_MAP[wt]
            configs[wt.value] = tuner(1.0)

        # Summary table
        print(f"{'Parameter':<25} {'Dialogue':>10} {'RAG':>10} {'Code':>10} {'Reasoning':>10}")
        print("─" * 70)
        for param, getter in [
            ("KV Budget", lambda c: f"{c.token_level.kv_cache_budget:.0%}"),
            ("Eviction Policy", lambda c: c.token_level.eviction_policy),
            ("Sink Tokens", lambda c: str(c.token_level.sink_token_count)),
            ("Sliding Window", lambda c: str(c.token_level.sliding_window_size)),
            ("Chunk Size", lambda c: str(c.model_level.chunk_size)),
            ("Chunked Eviction", lambda c: str(c.model_level.use_chunked_eviction)),
            ("Layer Strategy", lambda c: c.model_level.layer_budget_strategy),
            ("GPU Mem Util", lambda c: f"{c.system_level.gpu_memory_utilization:.0%}"),
            ("Batched Tokens", lambda c: str(c.system_level.max_num_batched_tokens)),
            ("Prefix Caching", lambda c: str(c.system_level.enable_prefix_caching)),
            ("Max Model Len", lambda c: str(c.system_level.max_model_len or "—")),
        ]:
            vals = [getter(configs[w]) for w in ["dialogue", "rag", "code", "reasoning"]]
            print(f"{param:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")
        return

    if args.prompt:
        from workload_classifier import classify_workload
        classification = classify_workload(args.prompt)
        config = generate_tuning_config(classification)
    elif args.workload:
        tuner = TUNING_MAP[WorkloadType(args.workload)]
        config = tuner(1.0)
    else:
        print("Use --workload, --prompt, or --compare-all")
        return

    print_tuning_config(config, args.model)
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        print(f"Config saved to: {args.output}")


if __name__ == '__main__':
    main()
