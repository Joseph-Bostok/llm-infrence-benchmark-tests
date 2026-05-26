#!/usr/bin/env python3
"""
PInsight Baseline Parameter Sweep

Systematically sweeps vLLM system-level parameters to establish
the baseline performance envelope. This is Phase 1 of the
parameter space exploration.

Parameter sweep values are sourced from parameter_space.py — the single
source of truth for all tunable parameters. Use --show-space to see the
full catalog before running a sweep.

Sweeps (SYSTEM / native tier from parameter_space.py):
  - gpu_memory_utilization: 0.80, 0.85, 0.90, 0.95
  - block_size: 8, 16, 32
  - max_model_len: 4096, 8192, 16384, 32768
  - enable_prefix_caching: True, False
  - max_num_batched_tokens: 2048, 4096, 8192, 16384
  - kv_cache_dtype: auto, fp8

Usage:
    # Show full parameter space catalog before sweeping:
    python3 baseline_sweep.py --show-space

    # Show native vLLM params only:
    python3 baseline_sweep.py --show-space --native-only

    # Full sweep (takes ~2-4 hours depending on GPU):
    python3 baseline_sweep.py --model Qwen/Qwen2.5-7B-Instruct

    # Quick sweep (fewer combos):
    python3 baseline_sweep.py --model Qwen/Qwen2.5-7B-Instruct --quick

    # Dry run (no GPU):
    python3 baseline_sweep.py --dry-run

    # Single config test:
    python3 baseline_sweep.py --model Qwen/Qwen2.5-7B-Instruct \\
        --gpu-mem 0.90 --block-size 16 --max-len 8192
"""

import json
import os
import sys
import time
import signal
import subprocess
import argparse
import itertools
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

# Import PInsight components
from vllm_kv_profiler import VLLMProfiler, load_prompts
from parameter_space import (
    PARAMETER_SPACE, Tier, print_table, print_algorithms, get_by_tier
)

RESULTS_DIR = Path(__file__).parent / "results" / "baseline_sweep"
WORKLOAD_DIR = Path(__file__).parent / "workloads"


@dataclass
class SweepConfig:
    """A single parameter configuration to test."""
    gpu_memory_utilization: float = 0.90
    block_size: int = 16
    max_model_len: int = 8192
    enable_prefix_caching: bool = False
    max_num_batched_tokens: int = 8192
    kv_cache_dtype: str = "auto"
    swap_space: int = 4

    def to_cli(self, model: str, port: int = 8000) -> str:
        parts = [
            "python3 -m vllm.entrypoints.openai.api_server",
            f"--model {model}",
            f"--port {port}",
            f"--gpu-memory-utilization {self.gpu_memory_utilization}",
            f"--block-size {self.block_size}",
            f"--max-model-len {self.max_model_len}",
            f"--max-num-batched-tokens {self.max_num_batched_tokens}",
        ]
        
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
            parts.append(f"--swap-space {self.swap_space}")
            
        parts.append(f"--kv-cache-dtype {self.kv_cache_dtype}")
        return " ".join(parts)

    def label(self) -> str:
        pc = "pc" if self.enable_prefix_caching else "nopc"
        return (f"mem{self.gpu_memory_utilization}_bs{self.block_size}_"
                f"len{self.max_model_len}_{pc}_bt{self.max_num_batched_tokens}_"
                f"{self.kv_cache_dtype}")


@dataclass
class SweepResult:
    """Results from a single sweep configuration."""
    config: Dict
    label: str
    workload: str
    avg_tps: float = 0.0
    avg_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    peak_kv_usage: float = 0.0
    total_tokens: int = 0
    num_requests: int = 0
    server_started: bool = True
    error: str = ""


def generate_sweep_configs(quick: bool = False) -> List[SweepConfig]:
    """Generate parameter combinations to sweep."""
    if quick:
        # Quick sweep: vary one param at a time from defaults
        configs = [SweepConfig()]  # default baseline
        for mem in [0.80, 0.85, 0.95]:
            configs.append(SweepConfig(gpu_memory_utilization=mem))
        for bs in [8, 32]:
            configs.append(SweepConfig(block_size=bs))
        for ml in [4096, 16384, 32768]:
            configs.append(SweepConfig(max_model_len=ml))
        configs.append(SweepConfig(enable_prefix_caching=True))
        for bt in [2048, 4096, 16384]:
            configs.append(SweepConfig(max_num_batched_tokens=bt))
        return configs
    else:
        # Full sweep: key parameter combinations
        configs = []
        for mem, bs, ml, pc in itertools.product(
            [0.85, 0.90, 0.95],       # gpu_memory_utilization
            [8, 16, 32],               # block_size
            [8192, 16384, 32768],      # max_model_len
            [False, True],             # prefix_caching
        ):
            configs.append(SweepConfig(
                gpu_memory_utilization=mem,
                block_size=bs,
                max_model_len=ml,
                enable_prefix_caching=pc,
            ))
        return configs


def start_vllm_server(config: SweepConfig, model: str, port: int = 8000,
                      dry_run: bool = False) -> Optional[subprocess.Popen]:
    """Start vLLM server with given config. Returns process handle."""
    if dry_run:
        return None

    cmd = config.to_cli(model, port)
    log_path = RESULTS_DIR / f"vllm_{config.label()}.log"
    log_file = open(log_path, 'w')

    try:
        proc = subprocess.Popen(
            cmd.split(), stdout=log_file, stderr=subprocess.STDOUT
        )
    except Exception as e:
        print(f"    Failed to start: {e}")
        return None

    # Wait for ready
    import requests
    for i in range(90):
        time.sleep(2)
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=2)
            if r.status_code == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            print(f"    Server exited early (code {proc.returncode})")
            return None

    print("    Server startup timeout")
    proc.terminate()
    return None


def stop_vllm_server(proc: Optional[subprocess.Popen]):
    """Stop vLLM server."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(2)


def run_sweep_config(config: SweepConfig, model: str, workload: str,
                     prompts: List[str], port: int = 8000,
                     dry_run: bool = False) -> SweepResult:
    """Run a single config and collect results."""
    result = SweepResult(
        config=asdict(config),
        label=config.label(),
        workload=workload,
    )

    if dry_run:
        profiler = VLLMProfiler(f"http://localhost:{port}", model=model, dry_run=True)
        report = profiler.run_workload(prompts[:3], max_tokens=64)
        result.avg_tps = report.avg_tps
        result.avg_ttft_ms = report.avg_ttft_ms
        result.p99_ttft_ms = report.p99_ttft_ms
        result.peak_kv_usage = report.peak_kv_cache_usage
        result.total_tokens = report.total_tokens
        result.num_requests = report.num_requests
        return result

    proc = start_vllm_server(config, model, port)
    if proc is None:
        result.server_started = False
        result.error = "Server failed to start (likely OOM or invalid config)"
        return result

    try:
        profiler = VLLMProfiler(f"http://localhost:{port}", model=model)
        if not profiler.check_server():
            result.error = "Server not reachable"
            return result

        report = profiler.run_workload(prompts, max_tokens=128)
        result.avg_tps = report.avg_tps
        result.avg_ttft_ms = report.avg_ttft_ms
        result.p99_ttft_ms = report.p99_ttft_ms
        result.peak_kv_usage = report.peak_kv_cache_usage
        result.total_tokens = report.total_tokens
        result.num_requests = report.num_requests
    except Exception as e:
        result.error = str(e)
    finally:
        stop_vllm_server(proc)

    return result


def print_results_table(results: List[SweepResult]):
    """Print results as a formatted table."""
    print(f"\n{'='*100}")
    print(f"BASELINE PARAMETER SWEEP RESULTS")
    print(f"{'='*100}")

    header = (f"{'GPU Mem':>7} {'BlkSz':>5} {'MaxLen':>7} {'PfxCch':>6} "
              f"{'BatTok':>6} {'Workload':>10} {'TPS':>7} {'TTFT':>8} "
              f"{'KV%':>6} {'Status':>8}")
    print(header)
    print("─" * 100)

    for r in results:
        c = r.config
        status = "✓" if r.server_started and not r.error else "✗"
        if r.error:
            status = r.error[:8]

        print(f"{c['gpu_memory_utilization']:>7.2f} "
              f"{c['block_size']:>5} "
              f"{c['max_model_len']:>7} "
              f"{'Yes' if c['enable_prefix_caching'] else 'No':>6} "
              f"{c['max_num_batched_tokens']:>6} "
              f"{r.workload:>10} "
              f"{r.avg_tps:>7.1f} "
              f"{r.avg_ttft_ms:>7.1f}ms "
              f"{r.peak_kv_usage:>5.1f}% "
              f"{status:>8}")

    print(f"{'='*100}")

    # Find best config per workload
    workloads = set(r.workload for r in results if r.server_started)
    for wl in sorted(workloads):
        wl_results = [r for r in results if r.workload == wl and r.avg_tps > 0]
        if wl_results:
            best = max(wl_results, key=lambda r: r.avg_tps)
            c = best.config
            print(f"\n  Best for {wl}: TPS={best.avg_tps:.1f}, "
                  f"mem={c['gpu_memory_utilization']}, "
                  f"block={c['block_size']}, "
                  f"len={c['max_model_len']}, "
                  f"prefix={'on' if c['enable_prefix_caching'] else 'off'}")


def main():
    parser = argparse.ArgumentParser(description="PInsight Baseline Parameter Sweep")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--quick", action="store_true",
                        help="Quick sweep (one-at-a-time, ~30 min)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock mode, no GPU needed")
    parser.add_argument("--workloads", type=str, nargs="+",
                        default=["dialogue", "rag", "code", "reasoning"])
    parser.add_argument("--output", type=str, default=None)
    # Parameter space inspection
    parser.add_argument("--show-space", action="store_true",
                        help="Print the full parameter space catalog and exit")
    parser.add_argument("--native-only", action="store_true",
                        help="With --show-space: show native vLLM params only")
    parser.add_argument("--show-algorithms", action="store_true",
                        help="Print all algorithms by category and exit")
    # Single config overrides
    parser.add_argument("--gpu-mem", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    args = parser.parse_args()

    # ── Parameter space inspection mode ──────────────────────────────────────
    if args.show_algorithms:
        print_algorithms()
        return

    if args.show_space:
        params = get_by_tier(Tier.NATIVE) if args.native_only else None
        print_table(params, show_notes=True)
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate configs
    if args.gpu_mem or args.block_size or args.max_len:
        configs = [SweepConfig(
            gpu_memory_utilization=args.gpu_mem or 0.90,
            block_size=args.block_size or 16,
            max_model_len=args.max_len or 8192,
        )]
    else:
        configs = generate_sweep_configs(quick=args.quick)

    print(f"PInsight Baseline Sweep")
    print(f"  Model: {args.model}")
    print(f"  Configs to test: {len(configs)}")
    print(f"  Workloads: {args.workloads}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Est. time: ~{len(configs) * len(args.workloads) * 3} min")
    print()

    all_results = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Config: {config.label()}")
        print(f"  CMD: {config.to_cli(args.model)[:80]}...")

        for wl in args.workloads:
            wl_path = WORKLOAD_DIR / f"{wl}.json"
            if not wl_path.exists():
                print(f"  Skipping {wl} — workload file missing")
                continue

            prompts = load_prompts(str(wl_path))
            print(f"  Running {wl} ({len(prompts)} prompts)...", end="", flush=True)

            result = run_sweep_config(
                config, args.model, wl, prompts,
                port=args.port, dry_run=args.dry_run
            )
            all_results.append(result)

            if result.error:
                print(f" FAILED: {result.error}")
            else:
                print(f" TPS={result.avg_tps:.1f}, "
                      f"TTFT={result.avg_ttft_ms:.0f}ms, "
                      f"KV={result.peak_kv_usage:.0f}%")

    # Print summary
    print_results_table(all_results)

    # Save results
    output_path = args.output or str(RESULTS_DIR / "sweep_results.json")
    with open(output_path, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Save as CSV for easy viewing
    csv_path = output_path.replace('.json', '.csv')
    with open(csv_path, 'w') as f:
        f.write("gpu_mem,block_size,max_len,prefix_cache,batched_tokens,"
                "workload,avg_tps,avg_ttft_ms,p99_ttft_ms,peak_kv_pct,status\n")
        for r in all_results:
            c = r.config
            status = "ok" if r.server_started and not r.error else r.error[:20]
            f.write(f"{c['gpu_memory_utilization']},{c['block_size']},"
                    f"{c['max_model_len']},{c['enable_prefix_caching']},"
                    f"{c['max_num_batched_tokens']},{r.workload},"
                    f"{r.avg_tps:.1f},{r.avg_ttft_ms:.1f},{r.p99_ttft_ms:.1f},"
                    f"{r.peak_kv_usage:.1f},{status}\n")
    print(f"CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
