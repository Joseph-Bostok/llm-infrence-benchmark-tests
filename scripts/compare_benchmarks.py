#!/usr/bin/env python3
"""
Benchmark Comparison Script
Compares Ollama, Etalon, and GuideLLM benchmark results.

Metrics compared:
- Time to First Token (TTFT)
- Time Per Output Token (TPOT) / Time Between Tokens (TBT)
- Inter-Token Latency (ITL) statistics
- Tokens per second throughput
- Fluidity metrics (Etalon-specific)
- GuideLLM sweep profiles and multimodal results
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


def load_guidellm_results(output_dir: str) -> Dict:
    """Load results from GuideLLM output directory."""
    output_path = Path(output_dir)

    # Try normalized results first
    normalized = output_path / "normalized_results.json"
    if normalized.exists():
        with open(normalized) as f:
            return json.load(f)

    # Fall back to benchmarks.json
    benchmarks = output_path / "benchmarks.json"
    if benchmarks.exists():
        with open(benchmarks) as f:
            return json.load(f)

    # Try any JSON file
    for results_file in output_path.glob("*.json"):
        with open(results_file) as f:
            return json.load(f)

    return {}


def extract_guidellm_metrics(data: Dict) -> Dict:
    """Extract key metrics from GuideLLM results."""
    metrics = {
        "source": "guidellm",
        "ttft": {},
        "itl": {},
        "tpot": {},
        "throughput": {},
    }

    # Handle normalized format from our wrapper
    if "metrics" in data:
        m = data["metrics"]
        metrics["ttft"] = m.get("ttft", {})
        metrics["itl"] = m.get("itl", {})
        metrics["tpot"] = m.get("tpot", {})
        metrics["throughput"] = m.get("throughput", {})
    # Handle raw GuideLLM format
    elif "benchmarks" in data:
        benchmarks = data["benchmarks"]
        if benchmarks:
            bench = benchmarks[-1]  # Last benchmark in sweep
            stats = bench.get("stats", bench.get("statistics", {}))
            raw_metrics = bench.get("metrics", {})

            ttft = stats.get("ttft", stats.get("time_to_first_token", {}))
            if not ttft and raw_metrics:
                ttft = raw_metrics.get("time_to_first_token_ms", {}).get("successful", {})
            
            if isinstance(ttft, dict):
                metrics["ttft"] = {
                    "mean": ttft.get("mean", ttft.get("avg")),
                    "p50": ttft.get("p50", ttft.get("median")),
                    "p90": ttft.get("p90"),
                    "p99": ttft.get("p99"),
                }
                # Handle nested percentiles
                if "percentiles" in ttft:
                    metrics["ttft"]["p50"] = ttft["percentiles"].get("p50")
                    metrics["ttft"]["p90"] = ttft["percentiles"].get("p90")
                    metrics["ttft"]["p99"] = ttft["percentiles"].get("p99")

            itl = stats.get("itl", stats.get("inter_token_latency", {}))
            if not itl and raw_metrics:
                itl = raw_metrics.get("inter_token_latency_ms", {}).get("successful", {})

            if isinstance(itl, dict):
                metrics["itl"] = {
                    "mean": itl.get("mean", itl.get("avg")),
                    "p50": itl.get("p50"),
                    "p90": itl.get("p90"),
                    "p99": itl.get("p99"),
                }
                # Handle nested percentiles
                if "percentiles" in itl:
                    metrics["itl"]["p50"] = itl["percentiles"].get("p50")
                    metrics["itl"]["p90"] = itl["percentiles"].get("p90")
                    metrics["itl"]["p99"] = itl["percentiles"].get("p99")

            tp = stats.get("throughput", {})
            if not tp and raw_metrics:
                tps = raw_metrics.get("output_tokens_per_second", {}).get("successful", {})
                if tps:
                    metrics["throughput"] = {
                        "tokens_per_second": tps.get("mean")
                    }

            if tp and isinstance(tp, dict):
                metrics["throughput"] = {
                    "tokens_per_second": tp.get("output_tokens_per_second",
                                                tp.get("tokens_per_second")),
                }

    return metrics


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


def compare_metrics(ollama_metrics: Dict, etalon_metrics: Dict, guidellm_metrics: Optional[Dict] = None) -> Dict:
    """Compare metrics between Ollama, Etalon, and GuideLLM benchmarks."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "ollama": ollama_metrics,
        "etalon": etalon_metrics,
        "guidellm": guidellm_metrics or {},
        "comparison": {}
    }

    # --- TTFT Comparison ---
    ttft_comp = {}

    # Ollama TTFT
    if ollama_metrics.get("ttft"):
        ollama_ttft = ollama_metrics["ttft"].get("mean", 0)
        if ollama_ttft:
            ttft_comp["ollama_ms"] = ollama_ttft * 1000

    # Etalon TTFT
    if etalon_metrics.get("ttft"):
        etalon_ttft_data = etalon_metrics["ttft"]
        etalon_ttft = 0
        for key, value in etalon_ttft_data.items():
            if "mean" in key.lower() and isinstance(value, (int, float)):
                etalon_ttft = value
                break
        if etalon_ttft:
            ttft_comp["etalon_ms"] = etalon_ttft * 1000

    # GuideLLM TTFT
    if guidellm_metrics and guidellm_metrics.get("ttft"):
        guidellm_ttft = guidellm_metrics["ttft"].get("mean")
        if guidellm_ttft:
            # GuideLLM may report in seconds or ms depending on version
            ttft_comp["guidellm_ms"] = guidellm_ttft * 1000 if guidellm_ttft < 10 else guidellm_ttft

    if len(ttft_comp) >= 2:
        comparison["comparison"]["ttft"] = ttft_comp

    # --- TPOT / TBT / ITL Comparison ---
    tpot_comp = {}

    if ollama_metrics.get("tpot"):
        ollama_tpot = ollama_metrics["tpot"].get("mean", 0)
        if ollama_tpot:
            tpot_comp["ollama_tpot_ms"] = ollama_tpot * 1000

    if etalon_metrics.get("tbt"):
        etalon_tbt_data = etalon_metrics["tbt"]
        etalon_tbt = 0
        for key, value in etalon_tbt_data.items():
            if "mean" in key.lower() and isinstance(value, (int, float)):
                etalon_tbt = value
                break
        if etalon_tbt:
            tpot_comp["etalon_tbt_ms"] = etalon_tbt * 1000

    if guidellm_metrics and guidellm_metrics.get("itl"):
        guidellm_itl = guidellm_metrics["itl"].get("mean")
        if guidellm_itl:
            tpot_comp["guidellm_itl_ms"] = guidellm_itl * 1000 if guidellm_itl < 10 else guidellm_itl

    if len(tpot_comp) >= 2:
        tpot_comp["note"] = "TPOT (Ollama) vs TBT (Etalon) vs ITL (GuideLLM) - similar but differ in measurement methodology"
        comparison["comparison"]["tpot_tbt_itl"] = tpot_comp

    # --- Throughput Comparison ---
    tp_comp = {}
    if ollama_metrics.get("throughput"):
        tp = ollama_metrics["throughput"].get("tokens_per_second_mean", 0)
        if tp:
            tp_comp["ollama_tok_s"] = tp

    if guidellm_metrics and guidellm_metrics.get("throughput"):
        tp = guidellm_metrics["throughput"].get("tokens_per_second")
        if tp:
            tp_comp["guidellm_tok_s"] = tp

    if tp_comp:
        comparison["comparison"]["throughput"] = tp_comp

    return comparison


def print_comparison(comparison: Dict) -> None:
    """Print formatted comparison results."""
    tools_present = []
    if comparison.get("ollama"): tools_present.append("Ollama")
    if comparison.get("etalon"): tools_present.append("Etalon")
    if comparison.get("guidellm"): tools_present.append("GuideLLM")

    print("\n" + "="*70)
    print(f"BENCHMARK COMPARISON: {' vs '.join(tools_present)}")
    print("="*70)

    ollama = comparison.get("ollama", {})
    etalon = comparison.get("etalon", {})
    guidellm = comparison.get("guidellm", {})
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

    # GuideLLM Summary
    print("\n--- GUIDELLM BENCHMARK ---")
    if guidellm:
        if "ttft" in guidellm and guidellm["ttft"]:
            ttft = guidellm["ttft"]
            mean_val = ttft.get('mean')
            if mean_val is not None:
                ms = mean_val * 1000 if mean_val < 10 else mean_val
                print(f"TTFT:  Mean={ms:.2f}ms")
                if ttft.get('p99'):
                    p99 = ttft['p99'] * 1000 if ttft['p99'] < 10 else ttft['p99']
                    print(f"       P99={p99:.2f}ms")
        if "itl" in guidellm and guidellm["itl"]:
            itl = guidellm["itl"]
            mean_val = itl.get('mean')
            if mean_val is not None:
                ms = mean_val * 1000 if mean_val < 10 else mean_val
                print(f"ITL:   Mean={ms:.2f}ms")
        if "throughput" in guidellm and guidellm["throughput"]:
            tp = guidellm["throughput"]
            if tp.get('tokens_per_second'):
                print(f"Throughput: {tp['tokens_per_second']:.2f} tok/s")
    else:
        print("No GuideLLM data available")

    # Direct Comparison
    print("\n--- DIRECT COMPARISON ---")
    if comp:
        if "ttft" in comp:
            c = comp["ttft"]
            print(f"\nTTFT (Time to First Token):")
            for key, val in c.items():
                if key.endswith('_ms') and isinstance(val, (int, float)):
                    label = key.replace('_ms', '').replace('_', ' ').title()
                    print(f"  {label}: {val:.2f} ms")

        if "tpot_tbt_itl" in comp:
            c = comp["tpot_tbt_itl"]
            print(f"\nTPOT/TBT/ITL (Token Generation Time):")
            for key, val in c.items():
                if key.endswith('_ms') and isinstance(val, (int, float)):
                    label = key.replace('_ms', '').replace('_', ' ').title()
                    print(f"  {label}: {val:.2f} ms")
            if c.get('note'):
                print(f"  Note: {c['note']}")

        if "throughput" in comp:
            c = comp["throughput"]
            print(f"\nThroughput (Tokens/s):")
            for key, val in c.items():
                if isinstance(val, (int, float)):
                    label = key.replace('_tok_s', '').replace('_', ' ').title()
                    print(f"  {label}: {val:.2f} tok/s")
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

    # Summary bar chart (three-way comparison)
    ax4 = axes[1, 1]
    comp = comparison.get("comparison", {})
    if comp:
        metrics_labels = []
        tool_data = {}  # tool_name -> [values]
        colors = {"Ollama": "#2196F3", "Etalon": "#FF9800", "GuideLLM": "#4CAF50"}

        if "ttft" in comp:
            metrics_labels.append("TTFT")
            c = comp["ttft"]
            for key, val in c.items():
                if key.endswith('_ms') and isinstance(val, (int, float)):
                    tool = key.replace('_ms', '').replace('_', ' ').title()
                    tool_data.setdefault(tool, []).append(val)

        tpot_key = "tpot_tbt_itl" if "tpot_tbt_itl" in comp else "tpot_tbt"
        if tpot_key in comp:
            metrics_labels.append("TPOT/TBT/ITL")
            c = comp[tpot_key]
            for key, val in c.items():
                if key.endswith('_ms') and isinstance(val, (int, float)):
                    tool = key.split('_')[0].title()
                    tool_data.setdefault(tool, []).append(val)

        if metrics_labels and tool_data:
            x = range(len(metrics_labels))
            n_tools = len(tool_data)
            width = 0.8 / max(n_tools, 1)

            for i, (tool, vals) in enumerate(tool_data.items()):
                offset = (i - n_tools/2 + 0.5) * width
                color = colors.get(tool, f"C{i}")
                ax4.bar([xi + offset for xi in x], vals[:len(metrics_labels)],
                        width, label=tool, color=color, alpha=0.85)

            ax4.set_xticks(list(x))
            ax4.set_xticklabels(metrics_labels)
            ax4.set_ylabel("Time (ms)")
            ax4.set_title("Three-Way Comparison")
            ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No comparison data\navailable", ha="center", va="center")
        ax4.set_title("Comparison")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare Ollama, Etalon, and GuideLLM benchmark results')
    parser.add_argument('--ollama-results', '-o', required=True,
                        help='Path to Ollama benchmark JSON results')
    parser.add_argument('--etalon-results', '-e',
                        help='Path to Etalon results directory')
    parser.add_argument('--guidellm-results', '-g',
                        help='Path to GuideLLM results directory')
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

    guidellm_metrics = {}
    if args.guidellm_results and Path(args.guidellm_results).exists():
        guidellm_data = load_guidellm_results(args.guidellm_results)
        guidellm_metrics = extract_guidellm_metrics(guidellm_data)

    # Compare
    comparison = compare_metrics(ollama_metrics, etalon_metrics, guidellm_metrics)

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
