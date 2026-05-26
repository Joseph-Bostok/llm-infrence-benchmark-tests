#!/usr/bin/env python3
"""
PInsight Paper Figure Generator

Generates publication-quality figures from experiment results:
    1. Tuning parameter comparison table (all workload types)
    2. Baseline vs tuned throughput bar chart
    3. KV cache utilization comparison
    4. Workload classification radar chart
    5. Multi-level optimization framework diagram

Usage:
    python visualizations/paper_figures.py --results results/experiments/
    python visualizations/paper_figures.py --demo  # Generate with sample data
"""

import json
import os
import sys
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install: pip install matplotlib")


# --- Color Palette ---
COLORS = {
    'baseline': '#6366f1',   # Indigo
    'tuned': '#06b6d4',      # Cyan
    'dialogue': '#f59e0b',   # Amber
    'rag': '#10b981',        # Emerald
    'code': '#8b5cf6',       # Violet
    'reasoning': '#ef4444',  # Red
    'bg': '#0f172a',         # Slate 900
    'text': '#e2e8f0',       # Slate 200
    'grid': '#334155',       # Slate 700
    'accent': '#38bdf8',     # Sky 400
}

WORKLOAD_NAMES = ['Dialogue', 'RAG', 'Code', 'Reasoning']
WORKLOAD_KEYS = ['dialogue', 'rag', 'code', 'reasoning']


def setup_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': '#1e293b',
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': COLORS['bg'],
    })


def load_comparison(results_dir: str) -> dict:
    """Load comparison results from experiment output."""
    comp_path = Path(results_dir) / "comparison.json"
    if comp_path.exists():
        with open(comp_path) as f:
            return json.load(f)
    return None


def get_demo_data() -> dict:
    """Generate demo data for testing without real experiments."""
    return {
        "workloads": [
            {"name": "dialogue", "baseline": {"avg_tps": 42.2, "avg_ttft_ms": 85.0, "peak_kv_cache_usage": 45.0}, "tuned": {"avg_tps": 52.8, "avg_ttft_ms": 62.0, "peak_kv_cache_usage": 28.0}},
            {"name": "rag", "baseline": {"avg_tps": 38.5, "avg_ttft_ms": 120.0, "peak_kv_cache_usage": 72.0}, "tuned": {"avg_tps": 45.1, "avg_ttft_ms": 95.0, "peak_kv_cache_usage": 58.0}},
            {"name": "code", "baseline": {"avg_tps": 40.0, "avg_ttft_ms": 95.0, "peak_kv_cache_usage": 55.0}, "tuned": {"avg_tps": 46.2, "avg_ttft_ms": 78.0, "peak_kv_cache_usage": 48.0}},
            {"name": "reasoning", "baseline": {"avg_tps": 35.0, "avg_ttft_ms": 140.0, "peak_kv_cache_usage": 68.0}, "tuned": {"avg_tps": 48.5, "avg_ttft_ms": 100.0, "peak_kv_cache_usage": 35.0}},
        ]
    }


def fig_throughput_comparison(data: dict, output_dir: str):
    """Bar chart comparing baseline vs tuned throughput per workload."""
    fig, ax = plt.subplots(figsize=(10, 6))

    workloads = data['workloads']
    x = np.arange(len(workloads))
    width = 0.35

    baseline_tps = [w['baseline'].get('avg_tps', 0) for w in workloads]
    tuned_tps = [w['tuned'].get('avg_tps', 0) for w in workloads]

    bars1 = ax.bar(x - width/2, baseline_tps, width, label='Baseline vLLM',
                   color=COLORS['baseline'], edgecolor='white', linewidth=0.5, alpha=0.9)
    bars2 = ax.bar(x + width/2, tuned_tps, width, label='PInsight-Tuned',
                   color=COLORS['tuned'], edgecolor='white', linewidth=0.5, alpha=0.9)

    # Add improvement labels
    for i, (b, t) in enumerate(zip(baseline_tps, tuned_tps)):
        if b > 0:
            pct = ((t - b) / b) * 100
            ax.annotate(f'+{pct:.0f}%', xy=(x[i] + width/2, t),
                       xytext=(0, 8), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold',
                       color=COLORS['accent'])

    ax.set_xlabel('Workload Type')
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_title('Baseline vs PInsight-Tuned Throughput')
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOAD_NAMES)
    ax.legend(loc='upper left', framealpha=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(baseline_tps), max(tuned_tps)) * 1.25)

    path = os.path.join(output_dir, 'fig_throughput_comparison.png')
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def fig_ttft_comparison(data: dict, output_dir: str):
    """Bar chart comparing TTFT (lower is better)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    workloads = data['workloads']
    x = np.arange(len(workloads))
    width = 0.35

    baseline = [w['baseline'].get('avg_ttft_ms', 0) for w in workloads]
    tuned = [w['tuned'].get('avg_ttft_ms', 0) for w in workloads]

    ax.bar(x - width/2, baseline, width, label='Baseline vLLM',
           color=COLORS['baseline'], edgecolor='white', linewidth=0.5, alpha=0.9)
    ax.bar(x + width/2, tuned, width, label='PInsight-Tuned',
           color=COLORS['tuned'], edgecolor='white', linewidth=0.5, alpha=0.9)

    for i, (b, t) in enumerate(zip(baseline, tuned)):
        if b > 0:
            pct = ((t - b) / b) * 100
            label = f'{pct:+.0f}%'
            color = '#22c55e' if pct < 0 else '#ef4444'
            ax.annotate(label, xy=(x[i] + width/2, t),
                       xytext=(0, 8), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold', color=color)

    ax.set_xlabel('Workload Type')
    ax.set_ylabel('Time to First Token (ms)')
    ax.set_title('TTFT Comparison (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOAD_NAMES)
    ax.legend(loc='upper left', framealpha=0.8)
    ax.grid(axis='y', alpha=0.3)

    path = os.path.join(output_dir, 'fig_ttft_comparison.png')
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def fig_kv_cache_usage(data: dict, output_dir: str):
    """KV cache peak usage comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    workloads = data['workloads']
    x = np.arange(len(workloads))
    width = 0.35

    baseline = [w['baseline'].get('peak_kv_cache_usage', 0) for w in workloads]
    tuned = [w['tuned'].get('peak_kv_cache_usage', 0) for w in workloads]

    ax.bar(x - width/2, baseline, width, label='Baseline vLLM',
           color=COLORS['baseline'], edgecolor='white', linewidth=0.5, alpha=0.9)
    ax.bar(x + width/2, tuned, width, label='PInsight-Tuned',
           color=COLORS['tuned'], edgecolor='white', linewidth=0.5, alpha=0.9)

    for i, (b, t) in enumerate(zip(baseline, tuned)):
        if b > 0:
            pct = ((t - b) / b) * 100
            label = f'{pct:+.0f}%'
            color = '#22c55e' if pct < 0 else '#ef4444'
            ax.annotate(label, xy=(x[i] + width/2, t),
                       xytext=(0, 8), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold', color=color)

    ax.set_xlabel('Workload Type')
    ax.set_ylabel('Peak KV Cache Usage (%)')
    ax.set_title('KV Cache Memory Usage (lower = more efficient)')
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOAD_NAMES)
    ax.legend(loc='upper left', framealpha=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    path = os.path.join(output_dir, 'fig_kv_cache_usage.png')
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def fig_tuning_parameters(output_dir: str):
    """Visual comparison of tuning parameters across workloads."""
    params = {
        'KV Budget (%)':     [50, 75, 80, 40],
        'Sink Tokens':       [4, 4, 8, 4],
        'Window Size':       [512, 256, 512, 256],
        'Chunk Size':        [16, 32, 8, 16],
        'GPU Mem (%)':       [85, 92, 90, 88],
        'Batch Tokens (K)':  [4, 16, 8, 8],
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    workload_colors = [COLORS[k] for k in WORKLOAD_KEYS]

    for idx, (param_name, values) in enumerate(params.items()):
        ax = axes[idx]
        bars = ax.bar(WORKLOAD_NAMES, values, color=workload_colors,
                      edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.set_title(param_name, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color=COLORS['accent'])
        ax.tick_params(axis='x', rotation=30, labelsize=9)

    fig.suptitle('PInsight Multi-Level Tuning Parameters by Workload',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, 'fig_tuning_parameters.png')
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def fig_optimization_framework(output_dir: str):
    """Diagram of the multi-level optimization framework."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(6, 6.7, 'PInsight Multi-Level KV Cache Optimization',
            ha='center', fontsize=16, fontweight='bold', color=COLORS['accent'])

    # Three level boxes
    levels = [
        (1, 4.5, 3, 1.5, 'Token Level', COLORS['dialogue'],
         ['KV Pruning (H2O/SnapKV)', 'Heavy-hitter detection', 'Attention sink preservation']),
        (4.5, 4.5, 3, 1.5, 'Model Level', COLORS['rag'],
         ['Chunking granularity', 'Layer-wise budgets', 'Index reuse (ChunkKV)']),
        (8, 4.5, 3, 1.5, 'System Level', COLORS['code'],
         ['Eviction algorithm', 'PagedAttention config', 'Memory offloading']),
    ]

    for x, y, w, h, title, color, items in levels:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.2, title, ha='center', va='top',
                fontsize=12, fontweight='bold', color=color)
        for i, item in enumerate(items):
            ax.text(x + 0.2, y + h - 0.5 - i*0.3, f'• {item}',
                    fontsize=8, color=COLORS['text'])

    # Feedback loop
    loop_y = 2.5
    loop_items = [
        (1.5, '① Profile'),
        (4, '② Classify'),
        (6.5, '③ Tune'),
        (9, '④ Monitor'),
    ]
    for lx, label in loop_items:
        rect = mpatches.FancyBboxPatch((lx, loop_y), 2, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor=COLORS['tuned'], alpha=0.3,
                                        edgecolor=COLORS['tuned'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(lx + 1, loop_y + 0.4, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color=COLORS['text'])

    # Arrows between loop items
    for i in range(len(loop_items) - 1):
        x1 = loop_items[i][0] + 2
        x2 = loop_items[i+1][0]
        ax.annotate('', xy=(x2, loop_y + 0.4), xytext=(x1, loop_y + 0.4),
                    arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))

    # Loop-back arrow
    ax.annotate('', xy=(1.5, loop_y + 0.4), xytext=(11, loop_y + 0.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2,
                               connectionstyle='arc3,rad=-0.4'))

    # Arrows from loop to levels
    ax.annotate('', xy=(2.5, 4.5), xytext=(2.5, loop_y + 0.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['grid'], lw=1.5, ls='--'))
    ax.annotate('', xy=(6, 4.5), xytext=(7.5, loop_y + 0.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['grid'], lw=1.5, ls='--'))
    ax.annotate('', xy=(9.5, 4.5), xytext=(7.5, loop_y + 0.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['grid'], lw=1.5, ls='--'))

    # Bottom label
    ax.text(6, 1.5, 'In-Situ Feedback Loop on vLLM',
            ha='center', fontsize=11, style='italic', color=COLORS['grid'])

    path = os.path.join(output_dir, 'fig_optimization_framework.png')
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="PInsight Paper Figure Generator")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to experiment results directory")
    parser.add_argument("--output", type=str, default="visualizations/figures",
                        help="Output directory for figures")
    parser.add_argument("--demo", action="store_true",
                        help="Generate with sample data")
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib required. Install: pip install matplotlib numpy")
        sys.exit(1)

    setup_style()
    os.makedirs(args.output, exist_ok=True)

    if args.demo or not args.results:
        data = get_demo_data()
        print("Using demo data (run with --results for real data)\n")
    else:
        data = load_comparison(args.results)
        if not data:
            print(f"No comparison.json found in {args.results}, using demo data")
            data = get_demo_data()

    print("Generating figures...")
    fig_throughput_comparison(data, args.output)
    fig_ttft_comparison(data, args.output)
    fig_kv_cache_usage(data, args.output)
    fig_tuning_parameters(args.output)
    fig_optimization_framework(args.output)
    print(f"\nAll figures saved to: {args.output}/")


if __name__ == '__main__':
    main()
