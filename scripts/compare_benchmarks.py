#!/usr/bin/env python3
"""
Benchmark Comparison Script
Compares Ollama custom benchmark results with Etalon results.

Metrics compared:
- Time to First Token (TTFT)
- Time Per Output Token (TPOT) / Time Between Tokens (TBT)
- Inter-Token Latency (ITL) statistics
- Tokens per second throughput
- Fluidity metrics (Etalon-specific)
"""

import json
import argparse
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization disabled.")


def load_ollama_results(filepath: str) -> Dict:
    """Load results from Ollama benchmark JSON file."""
    with open(filepath) as f:
        return json.load(f)


def load_etalon_results(output_dir: str) -> Dict:
    """Load results from Etalon output directory."""
    results = {}
    output_path = Path(output_dir)

    for results_file in output_path.glob("*.json"):
        with open(results_file) as f:
            results[results_file.stem] = json.load(f)

    return results


def extract_ollama_metrics(data: Dict) -> Dict:
    """Extract key metrics from Ollama benchmark results."""
    metrics_list = data.get("metrics", [])

    if not metrics_list:
        return {}

    # Aggregate across all requests
    ttfts = [m["ttft"] for m in metrics_list if m.get("ttft")]
    tpots = [m["tpot"] for m in metrics_list if m.get("tpot")]
    itl_means = [m["itl_mean"] for m in metrics_list if m.get("itl_mean")]
    tps_values = [m["tokens_per_second"] for m in metrics_list if m.get("tokens_per_second")]
    total_tokens = [m["total_tokens"] for m in metrics_list if m.get("total_tokens")]

    return {
        "source": "ollama_benchmark",
        "num_requests": len(metrics_list),
        "ttft": {
            "mean": statistics.mean(ttfts) if ttfts else 0,
            "std": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
            "min": min(ttfts) if ttfts else 0,
            "max": max(ttfts) if ttfts else 0,
        },
        "tpot": {
            "mean": statistics.mean(tpots) if tpots else 0,
            "std": statistics.stdev(tpots) if len(tpots) > 1 else 0,
            "min": min(tpots) if tpots else 0,
            "max": max(tpots) if tpots else 0,
        },
        "itl": {
            "mean": statistics.mean(itl_means) if itl_means else 0,
            "std": statistics.stdev(itl_means) if len(itl_means) > 1 else 0,
        },
        "throughput": {
            "tokens_per_second_mean": statistics.mean(tps_values) if tps_values else 0,
            "total_tokens": sum(total_tokens) if total_tokens else 0,
        },
        "raw_ttfts": ttfts,
        "raw_tpots": tpots,
        "raw_itls": itl_means,
    }


def extract_etalon_metrics(data: Dict) -> Dict:
    """Extract key metrics from Etalon benchmark results."""
    # Etalon stores metrics differently, adapt as needed
    metrics = {
        "source": "etalon",
        "ttft": {},
        "tbt": {},  # TBT is Etalon's equivalent of TPOT/ITL
        "throughput": {},
        "fluidity": {},  # Etalon-specific metric
    }

    # Parse Etalon's output format
    for key, value in data.items():
        if "ttft" in key.lower():
            metrics["ttft"][key] = value
        elif "tbt" in key.lower() or "time_between" in key.lower():
            metrics["tbt"][key] = value
        elif "throughput" in key.lower() or "token" in key.lower():
            metrics["throughput"][key] = value
        elif "fluid" in key.lower():
            metrics["fluidity"][key] = value
        else:
            metrics[key] = value

    return metrics


def compare_metrics(ollama_metrics: Dict, etalon_metrics: Dict) -> Dict:
    """Compare metrics between Ollama and Etalon benchmarks."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "ollama": ollama_metrics,
        "etalon": etalon_metrics,
        "comparison": {}
    }

    # Compare TTFT
    if ollama_metrics.get("ttft") and etalon_metrics.get("ttft"):
        ollama_ttft = ollama_metrics["ttft"].get("mean", 0)
        etalon_ttft_data = etalon_metrics["ttft"]

        # Find mean TTFT in Etalon data
        etalon_ttft = 0
        for key, value in etalon_ttft_data.items():
            if "mean" in key.lower() and isinstance(value, (int, float)):
                etalon_ttft = value
                break

        if ollama_ttft and etalon_ttft:
            diff = ollama_ttft - etalon_ttft
            diff_pct = (diff / etalon_ttft) * 100 if etalon_ttft else 0
            comparison["comparison"]["ttft"] = {
                "ollama_ms": ollama_ttft * 1000,
                "etalon_ms": etalon_ttft * 1000,
                "difference_ms": diff * 1000,
                "difference_pct": diff_pct,
            }

    # Compare TPOT/TBT
    if ollama_metrics.get("tpot") and etalon_metrics.get("tbt"):
        ollama_tpot = ollama_metrics["tpot"].get("mean", 0)
        etalon_tbt_data = etalon_metrics["tbt"]

        etalon_tbt = 0
        for key, value in etalon_tbt_data.items():
            if "mean" in key.lower() and isinstance(value, (int, float)):
                etalon_tbt = value
                break

        if ollama_tpot and etalon_tbt:
            diff = ollama_tpot - etalon_tbt
            diff_pct = (diff / etalon_tbt) * 100 if etalon_tbt else 0
            comparison["comparison"]["tpot_tbt"] = {
                "ollama_tpot_ms": ollama_tpot * 1000,
                "etalon_tbt_ms": etalon_tbt * 1000,
                "difference_ms": diff * 1000,
                "difference_pct": diff_pct,
                "note": "TPOT (Ollama) vs TBT (Etalon) - similar but may differ in calculation"
            }

    return comparison


def print_comparison(comparison: Dict) -> None:
    """Print formatted comparison results."""
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON: Ollama vs Etalon")
    print("="*70)

    ollama = comparison.get("ollama", {})
    etalon = comparison.get("etalon", {})
    comp = comparison.get("comparison", {})

    # Ollama Summary
    print("\n--- OLLAMA BENCHMARK ---")
    if ollama:
        print(f"Requests: {ollama.get('num_requests', 'N/A')}")
        if "ttft" in ollama:
            print(f"TTFT:  Mean={ollama['ttft']['mean']*1000:.2f}ms, "
                  f"Std={ollama['ttft']['std']*1000:.2f}ms, "
                  f"Min={ollama['ttft']['min']*1000:.2f}ms, "
                  f"Max={ollama['ttft']['max']*1000:.2f}ms")
        if "tpot" in ollama:
            print(f"TPOT:  Mean={ollama['tpot']['mean']*1000:.2f}ms, "
                  f"Std={ollama['tpot']['std']*1000:.2f}ms")
        if "itl" in ollama:
            print(f"ITL:   Mean={ollama['itl']['mean']*1000:.2f}ms, "
                  f"Std={ollama['itl']['std']*1000:.2f}ms")
        if "throughput" in ollama:
            print(f"Throughput: {ollama['throughput']['tokens_per_second_mean']:.2f} tok/s")
    else:
        print("No Ollama data available")

    # Etalon Summary
    print("\n--- ETALON BENCHMARK ---")
    if etalon:
        for category, data in etalon.items():
            if category == "source":
                continue
            if isinstance(data, dict) and data:
                print(f"\n{category.upper()}:")
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
    else:
        print("No Etalon data available")

    # Direct Comparison
    print("\n--- DIRECT COMPARISON ---")
    if comp:
        if "ttft" in comp:
            c = comp["ttft"]
            print(f"\nTTFT (Time to First Token):")
            print(f"  Ollama: {c['ollama_ms']:.2f} ms")
            print(f"  Etalon: {c['etalon_ms']:.2f} ms")
            print(f"  Difference: {c['difference_ms']:.2f} ms ({c['difference_pct']:+.1f}%)")

        if "tpot_tbt" in comp:
            c = comp["tpot_tbt"]
            print(f"\nTPOT/TBT (Token Generation Time):")
            print(f"  Ollama (TPOT): {c['ollama_tpot_ms']:.2f} ms")
            print(f"  Etalon (TBT):  {c['etalon_tbt_ms']:.2f} ms")
            print(f"  Difference: {c['difference_ms']:.2f} ms ({c['difference_pct']:+.1f}%)")
            print(f"  Note: {c.get('note', '')}")
    else:
        print("Insufficient data for direct comparison")

    print("\n" + "="*70)


def create_visualization(comparison: Dict, output_path: str) -> None:
    """Create visualization comparing the benchmarks."""
    if not HAS_MATPLOTLIB:
        print("Skipping visualization (matplotlib not available)")
        return

    ollama = comparison.get("ollama", {})

    if not ollama.get("raw_ttfts"):
        print("No raw data available for visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("LLM Inference Benchmark Comparison", fontsize=14)

    # TTFT Distribution
    ax1 = axes[0, 0]
    if ollama.get("raw_ttfts"):
        ttfts_ms = [t * 1000 for t in ollama["raw_ttfts"]]
        ax1.hist(ttfts_ms, bins=20, alpha=0.7, label="Ollama", color="blue")
        ax1.axvline(statistics.mean(ttfts_ms), color="red", linestyle="--",
                    label=f"Mean: {statistics.mean(ttfts_ms):.2f}ms")
    ax1.set_xlabel("Time to First Token (ms)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("TTFT Distribution")
    ax1.legend()

    # TPOT Distribution
    ax2 = axes[0, 1]
    if ollama.get("raw_tpots"):
        tpots_ms = [t * 1000 for t in ollama["raw_tpots"]]
        ax2.hist(tpots_ms, bins=20, alpha=0.7, label="Ollama", color="green")
        ax2.axvline(statistics.mean(tpots_ms), color="red", linestyle="--",
                    label=f"Mean: {statistics.mean(tpots_ms):.2f}ms")
    ax2.set_xlabel("Time Per Output Token (ms)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("TPOT Distribution")
    ax2.legend()

    # ITL over requests
    ax3 = axes[1, 0]
    if ollama.get("raw_itls"):
        itls_ms = [t * 1000 for t in ollama["raw_itls"]]
        ax3.plot(itls_ms, marker="o", linestyle="-", alpha=0.7)
        ax3.axhline(statistics.mean(itls_ms), color="red", linestyle="--",
                    label=f"Mean: {statistics.mean(itls_ms):.2f}ms")
    ax3.set_xlabel("Request #")
    ax3.set_ylabel("Mean Inter-Token Latency (ms)")
    ax3.set_title("ITL Across Requests")
    ax3.legend()

    # Summary bar chart
    ax4 = axes[1, 1]
    comp = comparison.get("comparison", {})
    if comp:
        metrics = []
        ollama_vals = []
        etalon_vals = []

        if "ttft" in comp:
            metrics.append("TTFT")
            ollama_vals.append(comp["ttft"]["ollama_ms"])
            etalon_vals.append(comp["ttft"]["etalon_ms"])

        if "tpot_tbt" in comp:
            metrics.append("TPOT/TBT")
            ollama_vals.append(comp["tpot_tbt"]["ollama_tpot_ms"])
            etalon_vals.append(comp["tpot_tbt"]["etalon_tbt_ms"])

        if metrics:
            x = range(len(metrics))
            width = 0.35
            ax4.bar([i - width/2 for i in x], ollama_vals, width, label="Ollama", color="blue")
            ax4.bar([i + width/2 for i in x], etalon_vals, width, label="Etalon", color="orange")
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics)
            ax4.set_ylabel("Time (ms)")
            ax4.set_title("Ollama vs Etalon Comparison")
            ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No comparison data\navailable", ha="center", va="center")
        ax4.set_title("Comparison")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare Ollama and Etalon benchmark results')
    parser.add_argument('--ollama-results', '-o', required=True,
                        help='Path to Ollama benchmark JSON results')
    parser.add_argument('--etalon-results', '-e',
                        help='Path to Etalon results directory')
    parser.add_argument('--output', type=str, default='./results/comparison.json',
                        help='Output path for comparison JSON')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--viz-output', type=str, default='./visualizations/comparison.png',
                        help='Output path for visualization')

    args = parser.parse_args()

    # Load results
    print("Loading benchmark results...")

    ollama_data = load_ollama_results(args.ollama_results)
    ollama_metrics = extract_ollama_metrics(ollama_data)

    etalon_metrics = {}
    if args.etalon_results and Path(args.etalon_results).exists():
        etalon_data = load_etalon_results(args.etalon_results)
        for key, data in etalon_data.items():
            etalon_metrics = extract_etalon_metrics(data)
            break  # Use first results file

    # Compare
    comparison = compare_metrics(ollama_metrics, etalon_metrics)

    # Print results
    print_comparison(comparison)

    # Save comparison
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        # Remove raw data for cleaner output
        save_comparison = comparison.copy()
        if "ollama" in save_comparison:
            save_comparison["ollama"] = {
                k: v for k, v in save_comparison["ollama"].items()
                if not k.startswith("raw_")
            }
        json.dump(save_comparison, f, indent=2)
    print(f"\nComparison saved to: {args.output}")

    # Visualize
    if args.visualize:
        Path(args.viz_output).parent.mkdir(parents=True, exist_ok=True)
        create_visualization(comparison, args.viz_output)


if __name__ == '__main__':
    main()
