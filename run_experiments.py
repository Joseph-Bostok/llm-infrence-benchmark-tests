#!/usr/bin/env python3
"""
PInsight Experiment Runner

Orchestrates the full experiment pipeline:
    1. Start vLLM server with baseline config
    2. Run workloads with vllm_kv_profiler
    3. Classify workloads
    4. Generate tuned config
    5. Restart vLLM with tuned config
    6. Re-run workloads
    7. Compare baseline vs tuned results
    8. Generate visualizations

Usage:
    # Full pipeline (requires GPU + vLLM):
    python run_experiments.py --model Qwen/Qwen2.5-7B-Instruct --gpu h100

    # Dry run (no GPU needed, tests the pipeline):
    python run_experiments.py --dry-run

    # Single phase:
    python run_experiments.py --phase baseline --dry-run
    python run_experiments.py --phase tuned --dry-run
    python run_experiments.py --phase compare --results-dir results/
"""

import json
import os
import sys
import time
import signal
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import PInsight components
from workload_classifier import classify_workload, WorkloadClassification
from tuning_engine import generate_tuning_config, TuningConfig, print_tuning_config
from vllm_kv_profiler import VLLMProfiler, ProfilingReport, load_prompts, print_report


WORKLOAD_DIR = Path(__file__).parent / "workloads"
RESULTS_DIR = Path(__file__).parent / "results" / "experiments"


def ensure_workloads_exist():
    """Check that workload files exist."""
    required = ["dialogue.json", "rag.json", "code.json", "reasoning.json"]
    missing = [f for f in required if not (WORKLOAD_DIR / f).exists()]
    if missing:
        print(f"Missing workload files: {missing}")
        print(f"Expected in: {WORKLOAD_DIR}")
        print("Run: python -c 'from run_experiments import create_workloads; create_workloads()'")
        return False
    return True


def create_workloads():
    """Create sample workload files if they don't exist."""
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)

    dialogue = [
        {"prompt": "System: You are a helpful coding assistant.\n\nUser: What is a KV cache in transformers?\n\nAssistant: A KV cache stores the key and value tensors from previous tokens during autoregressive generation, avoiding redundant computation.\n\nUser: Why does it become a bottleneck?\n\nAssistant: Because it grows linearly with sequence length. Every decode step must read the entire cache, consuming GPU memory bandwidth.\n\nUser: How can we reduce the KV cache size without losing accuracy?"},
        {"prompt": "System: You are a research advisor.\n\nUser: I'm working on LLM inference optimization.\n\nAssistant: That's a great area. What specific aspect?\n\nUser: KV cache compression. I've read about H2O and SnapKV.\n\nAssistant: Both are token eviction methods. H2O uses cumulative attention scores, SnapKV uses per-head observation windows.\n\nUser: Which one should I use for a chatbot application?\n\nAssistant: For chatbots, StreamingLLM's approach with attention sinks plus a sliding window works well.\n\nUser: What about for RAG workloads?\n\nAssistant: RAG is different — you need to preserve the retrieved document context. ChunkKV's chunk-level eviction is better there.\n\nUser: Can we automatically choose the right strategy based on the workload?"},
        {"prompt": "System: You are a technical support agent.\n\nUser: My vLLM server is running out of GPU memory with long conversations.\n\nAssistant: This is likely the KV cache growing too large. What's your max_model_len setting?\n\nUser: 32768 tokens.\n\nAssistant: At 32K tokens, the KV cache for a 7B model consumes about 1.9GB. Try reducing gpu_memory_utilization or enabling prefix caching.\n\nUser: I enabled prefix caching but it didn't help much.\n\nAssistant: Prefix caching helps when requests share common prefixes. For unique conversations, you may need to reduce max_model_len or use KV cache compression.\n\nUser: What compression options does vLLM support?"},
    ]

    rag = [
        {"prompt": "[Document 1] PagedAttention is a memory management technique introduced by vLLM that breaks the KV cache into fixed-size blocks. Unlike traditional implementations that pre-allocate contiguous memory for the maximum sequence length, PagedAttention allocates blocks on demand, similar to virtual memory paging in operating systems. This approach reduces memory waste from 60-80% to less than 4%, enabling significantly higher throughput.\n\n[Document 2] FlashAttention is an IO-aware exact attention algorithm that reduces memory reads by computing attention in tiles that fit in GPU SRAM. It avoids materializing the full N×N attention matrix in HBM, achieving O(n) memory complexity instead of O(n²). FlashAttention-2 further improves parallelism by partitioning across sequence length.\n\n[Document 3] H2O (Heavy-Hitter Oracle) identifies that only about 5% of tokens receive disproportionate attention scores. By retaining only these 'heavy hitters' plus a sliding window of recent tokens, H2O achieves up to 29× throughput improvement with minimal quality loss.\n\nBased on the documents above, explain how PagedAttention and H2O could be combined to optimize KV cache management."},
        {"prompt": "Context: The following are excerpts from recent research papers on KV cache optimization.\n\nSource 1: 'ChunkKV groups tokens into contiguous chunks and evaluates importance at the chunk level. This preserves linguistic coherence that per-token eviction destroys. On LongBench, ChunkKV outperforms H2O and SnapKV by up to 8.7% at equivalent compression ratios.'\n\nSource 2: 'KIVI introduces asymmetric quantization for KV caches. Keys exhibit outlier channels and are quantized per-channel, while values exhibit outlier tokens and are quantized per-token. This achieves 2-bit precision with near-baseline quality.'\n\nSource 3: 'MiniKV combines 2-bit quantization with token eviction using a layer-discriminative policy. The key insight is that more tokens at lower precision beats fewer tokens at higher precision.'\n\nQuestion: Compare and contrast the trade-offs between eviction-based approaches (ChunkKV) and quantization-based approaches (KIVI, MiniKV) for KV cache compression."},
        {"prompt": "[Document 1] The attention sink phenomenon, discovered by StreamingLLM, shows that the first few tokens in any sequence receive disproportionate attention regardless of semantic content. This is a mathematical artifact of softmax normalization.\n\n[Document 2] ShadowKV offloads value vectors to CPU memory while keeping compressed keys on GPU. During decode, it reconstructs only the sparse KV pairs needed.\n\n[Document 3] LMCache implements multi-tier cache sharing across GPU, CPU, and disk, enabling KV cache reuse across users and requests with shared prefixes.\n\n[Document 4] Mooncake disaggregates prefill and decode phases onto separate GPU pools, transferring KV cache between them.\n\nBased on these documents, describe a system architecture that combines multiple KV cache optimization strategies for a production LLM serving platform."},
    ]

    code = [
        {"prompt": "```python\nimport torch\nimport torch.nn as nn\nfrom typing import Optional, Tuple\n\nclass KVCache:\n    \"\"\"Manages key-value cache for transformer attention.\"\"\"\n    \n    def __init__(self, num_layers: int, num_heads: int, head_dim: int, \n                 max_seq_len: int, dtype=torch.float16):\n        self.num_layers = num_layers\n        self.num_heads = num_heads\n        self.head_dim = head_dim\n        self.max_seq_len = max_seq_len\n        self.keys = torch.zeros(num_layers, max_seq_len, num_heads, head_dim, dtype=dtype)\n        self.values = torch.zeros(num_layers, max_seq_len, num_heads, head_dim, dtype=dtype)\n        self.seq_len = 0\n    \n    def update(self, layer_idx: int, new_keys: torch.Tensor, new_values: torch.Tensor):\n        \"\"\"Append new KV pairs to the cache.\"\"\"\n        # TODO: implement update with bounds checking\n        pass\n    \n    def evict(self, keep_indices: torch.Tensor):\n        \"\"\"Evict tokens not in keep_indices.\"\"\"\n        # TODO: implement H2O-style eviction\n        pass\n```\n\nComplete the `update` and `evict` methods for this KV cache implementation. The update method should append new key-value pairs and handle the case where the cache is full. The evict method should retain only the tokens at the specified indices."},
        {"prompt": "```python\ndef compute_attention_importance(\n    attention_weights: torch.Tensor,  # shape: (num_layers, num_heads, seq_len)\n    method: str = 'h2o'\n) -> torch.Tensor:\n    \"\"\"\n    Compute token importance scores from attention weights.\n    \n    Args:\n        attention_weights: Cumulative attention weights per layer and head\n        method: 'h2o' for global scoring, 'snapkv' for per-head scoring\n    \n    Returns:\n        importance: Per-token importance scores, shape (seq_len,)\n    \"\"\"\n```\n\nImplement this function with both H2O and SnapKV scoring methods. Include attention sink detection (first N tokens with disproportionate scores)."},
        {"prompt": "I have a vLLM server running with this configuration:\n```python\nfrom vllm import LLM, SamplingParams\n\nllm = LLM(\n    model='Qwen/Qwen2.5-7B-Instruct',\n    gpu_memory_utilization=0.90,\n    max_model_len=32768,\n    block_size=16,\n    enable_prefix_caching=True,\n)\n```\n\nI need to add a monitoring hook that captures KV cache block utilization after each request. Write a Python class that wraps the vLLM engine and logs per-request KV cache metrics to a JSON file."},
    ]

    reasoning = [
        {"prompt": "Solve the following optimization problem step by step.\n\nA GPU has 80GB of HBM with 3.35 TB/s bandwidth. A 7B parameter model uses BF16 weights (15.24 GB). The KV cache uses 57 KB per token per layer, and the model has 28 layers with 4 KV heads.\n\nStep 1: Calculate the maximum batch size if we want to serve requests with 8192 token contexts.\nStep 2: Calculate the decode throughput (tokens/second) at this batch size.\nStep 3: If we apply 50% KV cache eviction, how does this change the maximum batch size?\nStep 4: What is the throughput improvement from eviction?\n\nLet's think through this carefully."},
        {"prompt": "Consider the following trade-off analysis.\n\nWe have three KV cache compression strategies:\n- Strategy A: H2O eviction at 50% budget (retains 82% attention mass)\n- Strategy B: ChunkKV at 50% budget with chunk_size=16 (retains 78% attention mass but preserves coherence)\n- Strategy C: KIVI 2-bit quantization (retains 100% tokens at 4x compression)\n\nFor each strategy, calculate:\n1. Memory savings\n2. Expected accuracy impact on LongBench\n3. Expected throughput improvement\n4. Which workloads each strategy is best suited for\n\nTherefore, step by step:\nFirst, let's establish the baseline memory usage...\nSecond, we need to consider the accuracy-memory trade-off curve...\nThird, the throughput improvement depends on whether we're memory-bound..."},
        {"prompt": "A data center runs 100 concurrent LLM inference sessions. Each session generates an average of 2000 tokens with a context window of 4096 tokens. The cluster has 8 H100 GPUs.\n\nStep 1: Calculate total KV cache memory needed.\nStep 2: Determine if this fits in GPU memory.\nStep 3: If not, propose an optimization strategy.\nStep 4: Calculate the expected throughput with and without optimization.\nStep 5: Determine the cost savings.\n\nLet's reason through this step by step."},
    ]

    for name, data in [("dialogue", dialogue), ("rag", rag),
                        ("code", code), ("reasoning", reasoning)]:
        path = WORKLOAD_DIR / f"{name}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Created {path}")


class ExperimentRunner:
    """Runs the full baseline vs. tuned experiment pipeline."""

    def __init__(self, model: str, server_url: str = "http://localhost:8000",
                 port: int = 8000, results_dir: str = None, dry_run: bool = False):
        self.model = model
        self.server_url = server_url
        self.port = port
        self.results_dir = Path(results_dir or RESULTS_DIR)
        self.dry_run = dry_run
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._vllm_proc = None

    def start_vllm(self, config: Optional[TuningConfig] = None, label: str = "baseline"):
        """Start a vLLM server with the given configuration."""
        if self.dry_run:
            print(f"[DRY RUN] Would start vLLM ({label})")
            return

        self.stop_vllm()

        if config:
            cmd = config.to_vllm_cli(self.model, self.port)
        else:
            cmd = (f"python -m vllm.entrypoints.openai.api_server "
                   f"--model {self.model} --port {self.port} "
                   f"--gpu-memory-utilization 0.90")

        print(f"\nStarting vLLM ({label}):")
        print(f"  {cmd}")

        log_path = self.results_dir / f"vllm_{label}.log"
        log_file = open(log_path, 'w')
        self._vllm_proc = subprocess.Popen(
            cmd.replace('\\\n    ', ' ').split(),
            stdout=log_file, stderr=subprocess.STDOUT
        )

        # Wait for server to be ready
        print("  Waiting for server...", end="", flush=True)
        for i in range(120):
            time.sleep(2)
            try:
                import requests
                r = requests.get(f"{self.server_url}/v1/models", timeout=2)
                if r.status_code == 200:
                    print(f" ready! (took {(i+1)*2}s)")
                    return
            except Exception:
                pass
            print(".", end="", flush=True)
        print(" TIMEOUT — check log at:", log_path)

    def stop_vllm(self):
        """Stop the vLLM server."""
        if self._vllm_proc:
            self._vllm_proc.terminate()
            self._vllm_proc.wait(timeout=10)
            self._vllm_proc = None

    def run_workload(self, workload_name: str, label: str = "baseline") -> Dict:
        """Run a single workload and collect results."""
        workload_path = WORKLOAD_DIR / f"{workload_name}.json"
        if not workload_path.exists():
            print(f"  Workload not found: {workload_path}")
            return {}

        prompts = load_prompts(str(workload_path))
        classification = classify_workload(prompts[0] if prompts else "")

        profiler = VLLMProfiler(self.server_url, model=self.model, dry_run=self.dry_run)
        if not self.dry_run:
            profiler.check_server()

        report = profiler.run_workload(prompts, max_tokens=128)
        report.workload_type = workload_name

        # Save individual report
        output_path = self.results_dir / f"{label}_{workload_name}.json"
        with open(output_path, 'w') as f:
            from dataclasses import asdict
            json.dump(asdict(report), f, indent=2)

        return {
            "workload": workload_name,
            "label": label,
            "classification": classification.workload_type.value,
            "confidence": classification.confidence,
            "avg_ttft_ms": report.avg_ttft_ms,
            "p50_ttft_ms": report.p50_ttft_ms,
            "p99_ttft_ms": report.p99_ttft_ms,
            "avg_tps": report.avg_tps,
            "total_tokens": report.total_tokens,
            "peak_kv_cache_usage": report.peak_kv_cache_usage,
            "avg_kv_cache_usage": report.avg_kv_cache_usage,
        }

    def run_baseline(self) -> List[Dict]:
        """Run all workloads with baseline vLLM config."""
        print("\n" + "=" * 60)
        print("PHASE 1: BASELINE (stock vLLM)")
        print("=" * 60)

        self.start_vllm(config=None, label="baseline")
        results = []
        for wl in ["dialogue", "rag", "code", "reasoning"]:
            print(f"\n--- {wl.upper()} workload ---")
            r = self.run_workload(wl, label="baseline")
            results.append(r)
        self.stop_vllm()
        return results

    def run_tuned(self) -> List[Dict]:
        """Run all workloads with PInsight-tuned vLLM configs."""
        print("\n" + "=" * 60)
        print("PHASE 2: PINSIGHT-TUNED (workload-adaptive)")
        print("=" * 60)

        results = []
        for wl in ["dialogue", "rag", "code", "reasoning"]:
            print(f"\n--- {wl.upper()} workload (tuned) ---")

            # Load a sample prompt for classification
            workload_path = WORKLOAD_DIR / f"{wl}.json"
            prompts = load_prompts(str(workload_path))
            classification = classify_workload(prompts[0] if prompts else "")
            config = generate_tuning_config(classification)

            print(f"  Classified as: {classification.workload_type.value}")
            print(f"  KV Budget: {config.token_level.kv_cache_budget:.0%}")
            print(f"  Eviction: {config.token_level.eviction_policy}")

            self.start_vllm(config=config, label=f"tuned_{wl}")
            r = self.run_workload(wl, label=f"tuned_{wl}")
            results.append(r)
            self.stop_vllm()

        return results

    def compare_results(self, baseline: List[Dict], tuned: List[Dict]) -> Dict:
        """Compare baseline vs tuned results."""
        print("\n" + "=" * 60)
        print("COMPARISON: BASELINE vs PINSIGHT-TUNED")
        print("=" * 60)

        comparison = {"timestamp": datetime.now().isoformat(), "workloads": []}

        print(f"\n{'Workload':<12} {'Metric':<15} {'Baseline':>10} {'Tuned':>10} {'Δ':>8}")
        print("─" * 60)

        for b, t in zip(baseline, tuned):
            wl = b.get("workload", "?")
            for metric in ["avg_ttft_ms", "avg_tps", "peak_kv_cache_usage"]:
                bv = b.get(metric, 0)
                tv = t.get(metric, 0)
                if bv > 0:
                    delta = ((tv - bv) / bv) * 100
                    sign = "+" if delta > 0 else ""
                    print(f"{wl:<12} {metric:<15} {bv:>10.1f} {tv:>10.1f} {sign}{delta:>6.1f}%")
                else:
                    print(f"{wl:<12} {metric:<15} {bv:>10.1f} {tv:>10.1f}     N/A")
                wl = ""  # Don't repeat workload name

            comparison["workloads"].append({
                "name": b.get("workload"),
                "baseline": b,
                "tuned": t,
            })
            print()

        # Save comparison
        comp_path = self.results_dir / "comparison.json"
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved to: {comp_path}")

        return comparison


def main():
    parser = argparse.ArgumentParser(description="PInsight Experiment Runner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "baseline", "tuned", "compare", "setup"],
                        help="Which phase to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with mock data (no GPU/vLLM needed)")
    args = parser.parse_args()

    runner = ExperimentRunner(
        model=args.model, server_url=args.server,
        port=args.port, results_dir=args.results_dir, dry_run=args.dry_run
    )

    if args.phase == "setup":
        print("Creating workload files...")
        create_workloads()
        print("\nShowing tuning configs for all workloads:")
        from tuning_engine import TUNING_MAP
        from workload_classifier import WorkloadType
        for wt in [WorkloadType.DIALOGUE, WorkloadType.RAG,
                    WorkloadType.CODE, WorkloadType.REASONING]:
            config = TUNING_MAP[wt](1.0)
            print(f"\n{'='*40}")
            print(f"  {wt.value.upper()}")
            print(f"  vLLM command: {config.to_vllm_cli(args.model)[:80]}...")
        return

    if not ensure_workloads_exist():
        print("\nCreating workload files first...")
        create_workloads()

    if args.phase in ("all", "baseline"):
        baseline = runner.run_baseline()
        bl_path = runner.results_dir / "baseline_summary.json"
        with open(bl_path, 'w') as f:
            json.dump(baseline, f, indent=2)
        print(f"\nBaseline results saved to: {bl_path}")
    else:
        bl_path = runner.results_dir / "baseline_summary.json"
        if bl_path.exists():
            with open(bl_path) as f:
                baseline = json.load(f)
        else:
            baseline = []

    if args.phase in ("all", "tuned"):
        tuned = runner.run_tuned()
        t_path = runner.results_dir / "tuned_summary.json"
        with open(t_path, 'w') as f:
            json.dump(tuned, f, indent=2)
        print(f"\nTuned results saved to: {t_path}")
    else:
        t_path = runner.results_dir / "tuned_summary.json"
        if t_path.exists():
            with open(t_path) as f:
                tuned = json.load(f)
        else:
            tuned = []

    if args.phase in ("all", "compare") and baseline and tuned:
        runner.compare_results(baseline, tuned)


if __name__ == '__main__':
    main()
