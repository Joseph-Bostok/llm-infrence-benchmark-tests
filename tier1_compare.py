#!/usr/bin/env python3
"""
Tier 1 Multi-Model Benchmark Comparison
========================================
Reads all benchmark results from tier1_benchmark.sh and produces:
  1. A formatted terminal comparison table
  2. A combined JSON file for further analysis / dashboard
  3. A CSV export for spreadsheets

Usage:
    python3 tier1_compare.py [results_dir]
    python3 tier1_compare.py ./results/tier1_7-8B/
"""

import json
import os
import sys
import csv
from pathlib import Path
from statistics import mean, stdev, median


def load_results(results_dir: str) -> list[dict]:
    """Load all model benchmark results from the tier directory."""
    models = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: {results_dir} does not exist")
        sys.exit(1)

    for model_dir in sorted(results_path.iterdir()):
        if not model_dir.is_dir():
            continue

        result_file = model_dir / "results.json"
        metadata_file = model_dir / "metadata.json"

        if not result_file.exists():
            print(f"  Skipping {model_dir.name}: no results.json")
            continue

        with open(result_file) as f:
            results = json.load(f)

        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

        models.append({
            "dir_name": model_dir.name,
            "metadata": metadata,
            "results": results
        })

    return models


def extract_metrics(model_data: dict) -> dict:
    """Extract key metrics from a model's benchmark results.

    Expected JSON structure from ollama_benchmark.py:
    {
      "timestamp": "...",
      "host": "...",
      "model": "llama3:latest",
      "metrics": [
        {
          "prompt": "...",
          "request_start": <monotonic_seconds>,
          "first_token_time": <monotonic_seconds>,
          "last_token_time": <monotonic_seconds>,
          "request_end": <monotonic_seconds>,
          "token_timings": [
            {"token_index": 0, "token": "...", "timestamp": ..., "latency": <seconds>},
            ...
          ]
        },
        ...
      ]
    }
    """
    results = model_data["results"]
    metadata = model_data.get("metadata", {})

    # Get the metrics array
    requests = results.get("metrics", [])
    if not requests:
        return None

    # Compute per-request metrics from raw timestamps
    ttfts = []       # Time to first token (ms)
    tpots = []       # Time per output token (ms)
    itl_means = []   # Mean inter-token latency per request (ms)
    itl_all = []     # All individual ITL values across all requests (ms)
    throughputs = []  # Tokens per second
    output_tokens = []  # Number of tokens per request
    total_times = []    # Total request time (ms)

    for req in requests:
        request_start = req.get("request_start", 0)
        first_token_time = req.get("first_token_time", 0)
        last_token_time = req.get("last_token_time", 0)
        request_end = req.get("request_end", 0)
        token_timings = req.get("token_timings", [])

        if not token_timings or first_token_time <= request_start:
            continue

        num_tokens = len(token_timings)
        output_tokens.append(num_tokens)

        # TTFT: time from request start to first token (convert s -> ms)
        ttft = (first_token_time - request_start) * 1000.0
        ttfts.append(ttft)

        # Total request time (s -> ms)
        total_ms = (request_end - request_start) * 1000.0
        total_times.append(total_ms)

        # Inter-token latencies: skip token 0 (that's the TTFT)
        # token_timings[i].latency is already the delta in seconds
        itls_this_request = []
        for t in token_timings[1:]:
            latency_ms = t["latency"] * 1000.0
            itls_this_request.append(latency_ms)
            itl_all.append(latency_ms)

        if itls_this_request:
            itl_means.append(mean(itls_this_request))

        # TPOT: total decode time / number of decode steps
        if num_tokens > 1:
            decode_time = (last_token_time - first_token_time) * 1000.0  # ms
            tpot = decode_time / (num_tokens - 1)
            tpots.append(tpot)

        # Throughput: tokens per second
        if last_token_time > first_token_time:
            decode_seconds = last_token_time - first_token_time
            tps = (num_tokens - 1) / decode_seconds
            throughputs.append(tps)

    if not ttfts:
        return None

    # Exclude first request (cold-start) for warm stats
    ttfts_warm = ttfts[1:] if len(ttfts) > 1 else ttfts

    def safe_stats(values):
        if not values:
            return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0,
                    "p50": 0, "p90": 0, "p99": 0, "count": 0}
        s = sorted(values)
        n = len(s)
        return {
            "mean": round(mean(s), 2),
            "median": round(median(s), 2),
            "std": round(stdev(s), 2) if n > 1 else 0,
            "min": round(min(s), 2),
            "max": round(max(s), 2),
            "p50": round(s[int(n * 0.50)], 2),
            "p90": round(s[min(int(n * 0.90), n - 1)], 2),
            "p99": round(s[min(int(n * 0.99), n - 1)], 2),
            "count": n,
        }

    return {
        "display_name": metadata.get("display_name", model_data["dir_name"]),
        "param_count": metadata.get("param_count", "?"),
        "architecture": metadata.get("architecture", "?"),
        "model_tag": metadata.get("model_tag", results.get("model", "?")),
        "num_requests": len(requests),
        "total_tokens": sum(output_tokens) if output_tokens else 0,
        "ttft_cold_ms": round(ttfts[0], 2),
        "ttft": safe_stats(ttfts_warm),
        "tpot": safe_stats(tpots),
        "itl": safe_stats(itl_means),
        "itl_all": safe_stats(itl_all),
        "throughput": safe_stats(throughputs),
        "output_tokens": safe_stats([float(x) for x in output_tokens]),
        "total_time": safe_stats(total_times),
        "elapsed_seconds": metadata.get("elapsed_seconds", 0),
    }


def print_comparison_table(all_metrics: list[dict]):
    """Print a formatted comparison table to terminal."""
    if not all_metrics:
        print("No metrics to compare.")
        return

    print()
    print("=" * 120)
    print(f"{'TIER 1 MODEL COMPARISON (7-8B)':^120}")
    print("=" * 120)
    print()

    # Main metrics table
    header = f"{'Model':<20} {'Params':>6} {'Reqs':>5} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'ITL(ms)':>10} {'Tok/s':>8} {'Tokens':>8} {'Cold(ms)':>10}"
    print(header)
    print("-" * len(header))

    for m in all_metrics:
        print(f"{m['display_name']:<20} "
              f"{m['param_count']:>6} "
              f"{m['num_requests']:>5} "
              f"{m['ttft']['mean']:>10.1f} "
              f"{m['tpot']['mean']:>10.2f} "
              f"{m['itl']['mean']:>10.2f} "
              f"{m['throughput']['mean']:>8.1f} "
              f"{m['total_tokens']:>8} "
              f"{m['ttft_cold_ms']:>10.1f}")

    print()

    # Percentile table
    print(f"{'Model':<20} {'TTFT p50':>10} {'TTFT p99':>10} {'ITL p50':>10} {'ITL p90':>10} {'ITL p99':>10} {'ITL std':>10} {'Tput min':>10} {'Tput max':>10}")
    print("-" * 110)

    for m in all_metrics:
        print(f"{m['display_name']:<20} "
              f"{m['ttft']['p50']:>10.1f} "
              f"{m['ttft']['p99']:>10.1f} "
              f"{m['itl']['p50']:>10.2f} "
              f"{m['itl']['p90']:>10.2f} "
              f"{m['itl']['p99']:>10.2f} "
              f"{m['itl']['std']:>10.2f} "
              f"{m['throughput']['min']:>10.1f} "
              f"{m['throughput']['max']:>10.1f}")

    print()

    # All-token ITL stats (granular)
    print(f"{'Model':<20} {'ITL tokens':>10} {'ITL p50':>10} {'ITL p90':>10} {'ITL p99':>10} {'ITL min':>10} {'ITL max':>10}")
    print("-" * 90)

    for m in all_metrics:
        itl = m['itl_all']
        print(f"{m['display_name']:<20} "
              f"{itl['count']:>10} "
              f"{itl['p50']:>10.2f} "
              f"{itl['p90']:>10.2f} "
              f"{itl['p99']:>10.2f} "
              f"{itl['min']:>10.2f} "
              f"{itl['max']:>10.2f}")

    print()

    # Rankings
    print("RANKINGS")
    print("-" * 60)

    by_ttft = sorted(all_metrics, key=lambda m: m["ttft"]["mean"])
    print(f"  Fastest TTFT (warm):    {by_ttft[0]['display_name']} ({by_ttft[0]['ttft']['mean']:.1f}ms)")

    by_itl = sorted(all_metrics, key=lambda m: m["itl"]["mean"])
    print(f"  Lowest ITL:             {by_itl[0]['display_name']} ({by_itl[0]['itl']['mean']:.2f}ms)")

    by_itl_std = sorted(all_metrics, key=lambda m: m["itl"]["std"])
    print(f"  Most consistent ITL:    {by_itl_std[0]['display_name']} (std={by_itl_std[0]['itl']['std']:.2f}ms)")

    by_tput = sorted(all_metrics, key=lambda m: m["throughput"]["mean"], reverse=True)
    print(f"  Highest throughput:     {by_tput[0]['display_name']} ({by_tput[0]['throughput']['mean']:.1f} tok/s)")

    by_cold = sorted(all_metrics, key=lambda m: m["ttft_cold_ms"])
    print(f"  Fastest cold start:     {by_cold[0]['display_name']} ({by_cold[0]['ttft_cold_ms']:.0f}ms)")

    by_tokens = sorted(all_metrics, key=lambda m: m["total_tokens"], reverse=True)
    print(f"  Most tokens generated:  {by_tokens[0]['display_name']} ({by_tokens[0]['total_tokens']})")

    print()


def save_combined_json(all_metrics: list[dict], output_path: str):
    """Save combined comparison data as JSON."""
    output = {
        "tier": "tier1_7-8B",
        "num_models": len(all_metrics),
        "models": all_metrics,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Combined JSON saved: {output_path}")


def save_csv(all_metrics: list[dict], output_path: str):
    """Save comparison as CSV for spreadsheets."""
    if not all_metrics:
        return

    fieldnames = [
        "model", "params", "architecture", "model_tag", "requests", "total_tokens",
        "ttft_cold_ms", "ttft_mean_ms", "ttft_p50_ms", "ttft_p99_ms",
        "tpot_mean_ms", "tpot_std_ms",
        "itl_mean_ms", "itl_p50_ms", "itl_p90_ms", "itl_p99_ms", "itl_std_ms",
        "itl_all_p50_ms", "itl_all_p99_ms", "itl_all_min_ms", "itl_all_max_ms",
        "throughput_mean", "throughput_min", "throughput_max",
        "avg_output_tokens",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({
                "model": m["display_name"],
                "params": m["param_count"],
                "architecture": m["architecture"],
                "model_tag": m["model_tag"],
                "requests": m["num_requests"],
                "total_tokens": m["total_tokens"],
                "ttft_cold_ms": m["ttft_cold_ms"],
                "ttft_mean_ms": m["ttft"]["mean"],
                "ttft_p50_ms": m["ttft"]["p50"],
                "ttft_p99_ms": m["ttft"]["p99"],
                "tpot_mean_ms": m["tpot"]["mean"],
                "tpot_std_ms": m["tpot"]["std"],
                "itl_mean_ms": m["itl"]["mean"],
                "itl_p50_ms": m["itl"]["p50"],
                "itl_p90_ms": m["itl"]["p90"],
                "itl_p99_ms": m["itl"]["p99"],
                "itl_std_ms": m["itl"]["std"],
                "itl_all_p50_ms": m["itl_all"]["p50"],
                "itl_all_p99_ms": m["itl_all"]["p99"],
                "itl_all_min_ms": m["itl_all"]["min"],
                "itl_all_max_ms": m["itl_all"]["max"],
                "throughput_mean": m["throughput"]["mean"],
                "throughput_min": m["throughput"]["min"],
                "throughput_max": m["throughput"]["max"],
                "avg_output_tokens": m["output_tokens"]["mean"],
            })
    print(f"CSV saved: {output_path}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./results/tier1_7-8B/"

    print(f"Loading results from: {results_dir}")
    models = load_results(results_dir)
    print(f"Found {len(models)} model result(s)")

    if not models:
        print("No results found. Run tier1_benchmark.sh first.")
        sys.exit(1)

    # Extract metrics
    all_metrics = []
    for model in models:
        metrics = extract_metrics(model)
        if metrics:
            all_metrics.append(metrics)
        else:
            print(f"  Warning: Could not extract metrics from {model['dir_name']}")

    # Print comparison
    print_comparison_table(all_metrics)

    # Save outputs
    output_dir = Path(results_dir)
    save_combined_json(all_metrics, str(output_dir / "tier1_comparison.json"))
    save_csv(all_metrics, str(output_dir / "tier1_comparison.csv"))


if __name__ == "__main__":
    main()
