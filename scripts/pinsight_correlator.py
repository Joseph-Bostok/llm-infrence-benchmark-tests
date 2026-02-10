#!/usr/bin/env python3
"""
PInsight Trace Correlator
Correlates PInsight/LTTng system-level traces with LLM inference metrics.

Maps between:
- Application-level: TTFT, TPOT, ITL, fluidity-index
- Runtime-level: CUDA kernels, memory transfers, Python events
- System-level: CPU scheduling, page faults, network I/O

This tool provides the bridge between PInsight's HPC tracing capabilities
and LLM inference benchmarking metrics, supporting the thesis that
system-level traces reveal bottlenecks invisible to application metrics alone.
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class TraceEvent:
    """A single trace event from any layer."""
    timestamp: float  # nanoseconds from epoch
    layer: str  # "application", "runtime", "system"
    event_type: str  # e.g., "token_generated", "cuda_kernel", "page_fault"
    duration_ns: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class CorrelationResult:
    """Result of correlating traces across layers."""
    token_index: int
    app_timestamp: float
    app_latency_ms: float
    # Runtime correlation
    cuda_kernel_name: Optional[str] = None
    cuda_kernel_duration_ms: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    # System correlation
    cpu_utilization: Optional[float] = None
    page_faults: Optional[int] = None
    context_switches: Optional[int] = None
    # Anomaly flags
    is_latency_spike: bool = False
    spike_cause: Optional[str] = None


@dataclass
class CorrelationReport:
    """Full cross-layer correlation report."""
    model: str
    timestamp: str
    total_tokens: int = 0
    # Application metrics
    ttft_ms: float = 0.0
    avg_itl_ms: float = 0.0
    tokens_per_second: float = 0.0
    # Runtime summary
    avg_kernel_duration_ms: float = 0.0
    gpu_utilization_pct: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    # System summary
    avg_cpu_utilization: float = 0.0
    total_page_faults: int = 0
    total_context_switches: int = 0
    # Correlations
    correlations: List[Dict] = field(default_factory=list)
    # Anomalies detected
    anomalies: List[Dict] = field(default_factory=list)
    # Insights
    insights: List[str] = field(default_factory=list)


def load_ollama_benchmark_results(filepath: str) -> Dict:
    """Load results from ollama_benchmark.py output."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_lttng_trace(trace_dir: str) -> List[TraceEvent]:
    """
    Load events from an LTTng trace directory.

    Expected trace format: CTF (Common Trace Format)
    Tools like babeltrace2 can convert CTF to JSON.

    For now, supports pre-converted JSON format:
    [{"timestamp": <ns>, "name": "<event>", "fields": {...}}, ...]
    """
    events = []

    # Look for babeltrace2 JSON output
    json_files = list(Path(trace_dir).glob("*.json"))
    if not json_files:
        # Try to convert CTF to JSON using babeltrace2
        ctf_path = Path(trace_dir)
        if ctf_path.exists():
            print(f"Note: No JSON trace found. To convert CTF traces, run:")
            print(f"  babeltrace2 --output-format=json {trace_dir} > {trace_dir}/trace.json")
            return events

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                raw_events = json.load(f)

            for raw in raw_events:
                event = TraceEvent(
                    timestamp=raw.get('timestamp', 0),
                    layer="system",
                    event_type=raw.get('name', raw.get('event', '')),
                    duration_ns=raw.get('duration', 0),
                    metadata=raw.get('fields', raw.get('metadata', {}))
                )
                events.append(event)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {json_file}: {e}")

    return sorted(events, key=lambda e: e.timestamp)


def load_nsight_trace(filepath: str) -> List[TraceEvent]:
    """
    Load CUDA kernel events from Nsight Systems / Nsight Compute output.

    Supports:
    - nsys export --type=json output
    - nv-nsight-cu-cli --csv output (converted to JSON)
    """
    events = []

    if not os.path.exists(filepath):
        return events

    try:
        with open(filepath, 'r') as f:
            raw = json.load(f)

        # Handle nsys JSON format
        kernel_events = raw if isinstance(raw, list) else raw.get('traceEvents', [])

        for entry in kernel_events:
            if entry.get('cat') in ('kernel', 'gpu', 'cuda'):
                event = TraceEvent(
                    timestamp=entry.get('ts', 0) * 1000,  # μs to ns
                    layer="runtime",
                    event_type=entry.get('name', 'unknown_kernel'),
                    duration_ns=entry.get('dur', 0) * 1000,
                    metadata={
                        'grid': entry.get('args', {}).get('grid', ''),
                        'block': entry.get('args', {}).get('block', ''),
                        'registers': entry.get('args', {}).get('registers', 0),
                        'shared_memory': entry.get('args', {}).get('shared_memory', 0),
                    }
                )
                events.append(event)

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not parse Nsight trace {filepath}: {e}")

    return sorted(events, key=lambda e: e.timestamp)


def extract_token_events(benchmark_results: Dict) -> List[TraceEvent]:
    """Extract token generation events from benchmark results as trace events."""
    events = []

    metrics_list = benchmark_results.get('metrics', [])
    for metrics in metrics_list:
        token_timings = metrics.get('token_timings', [])
        for timing in token_timings:
            event = TraceEvent(
                timestamp=timing['timestamp'] * 1e9,  # seconds to nanoseconds
                layer="application",
                event_type="token_generated",
                duration_ns=timing['latency'] * 1e9,
                metadata={
                    'token_index': timing['token_index'],
                    'token': timing['token'],
                    'latency_ms': timing['latency'] * 1000,
                }
            )
            events.append(event)

    return events


def correlate_traces(
    app_events: List[TraceEvent],
    runtime_events: List[TraceEvent],
    system_events: List[TraceEvent],
    window_ns: float = 5_000_000,  # 5ms correlation window
) -> List[CorrelationResult]:
    """
    Correlate events across application, runtime, and system layers.

    For each token generation event:
    1. Find overlapping CUDA kernels (within time window)
    2. Find concurrent system events (page faults, context switches)
    3. Flag anomalies where system events explain latency spikes
    """
    correlations = []

    # Compute ITL statistics for anomaly detection
    if app_events:
        latencies = [e.metadata.get('latency_ms', 0) for e in app_events]
        if latencies:
            mean_lat = statistics.mean(latencies)
            std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0
            spike_threshold = mean_lat + 3 * std_lat  # 3-sigma
        else:
            spike_threshold = float('inf')
    else:
        spike_threshold = float('inf')

    for app_event in app_events:
        token_ts = app_event.timestamp
        token_idx = app_event.metadata.get('token_index', -1)
        latency_ms = app_event.metadata.get('latency_ms', 0)

        result = CorrelationResult(
            token_index=token_idx,
            app_timestamp=token_ts,
            app_latency_ms=latency_ms,
        )

        # Find overlapping CUDA kernels
        for rt_event in runtime_events:
            rt_start = rt_event.timestamp
            rt_end = rt_start + rt_event.duration_ns
            if (rt_start - window_ns) <= token_ts <= (rt_end + window_ns):
                result.cuda_kernel_name = rt_event.event_type
                result.cuda_kernel_duration_ms = rt_event.duration_ns / 1e6
                break

        # Find concurrent system events
        page_faults = 0
        ctx_switches = 0
        for sys_event in system_events:
            if abs(sys_event.timestamp - token_ts) <= window_ns:
                if 'page_fault' in sys_event.event_type.lower():
                    page_faults += 1
                elif 'sched_switch' in sys_event.event_type.lower():
                    ctx_switches += 1

        result.page_faults = page_faults
        result.context_switches = ctx_switches

        # Anomaly detection
        if latency_ms > spike_threshold:
            result.is_latency_spike = True
            if page_faults > 0:
                result.spike_cause = f"page_faults ({page_faults})"
            elif ctx_switches > 5:
                result.spike_cause = f"context_switches ({ctx_switches})"
            elif result.cuda_kernel_duration_ms and result.cuda_kernel_duration_ms > latency_ms * 0.8:
                result.spike_cause = f"long_kernel ({result.cuda_kernel_name})"
            else:
                result.spike_cause = "unknown"

        correlations.append(result)

    return correlations


def generate_report(
    benchmark_results: Dict,
    correlations: List[CorrelationResult],
    model: str,
) -> CorrelationReport:
    """Generate a comprehensive correlation report."""
    report = CorrelationReport(
        model=model,
        timestamp=datetime.now().isoformat(),
    )

    # Application metrics
    metrics_list = benchmark_results.get('metrics', [])
    if metrics_list:
        m = metrics_list[0]  # Use first result
        report.ttft_ms = m.get('ttft', 0) * 1000
        report.avg_itl_ms = m.get('itl_mean', 0) * 1000
        report.tokens_per_second = m.get('tokens_per_second', 0)
        report.total_tokens = m.get('total_tokens', 0)

    # Correlation data
    report.correlations = [asdict(c) for c in correlations]

    # Anomalies
    anomalies = [c for c in correlations if c.is_latency_spike]
    report.anomalies = [asdict(a) for a in anomalies]

    # Aggregate system metrics
    if correlations:
        pf_counts = [c.page_faults for c in correlations if c.page_faults is not None]
        cs_counts = [c.context_switches for c in correlations if c.context_switches is not None]
        report.total_page_faults = sum(pf_counts)
        report.total_context_switches = sum(cs_counts)

        kernel_durations = [
            c.cuda_kernel_duration_ms for c in correlations
            if c.cuda_kernel_duration_ms is not None
        ]
        if kernel_durations:
            report.avg_kernel_duration_ms = statistics.mean(kernel_durations)

    # Generate insights
    report.insights = generate_insights(report, correlations)

    return report


def generate_insights(
    report: CorrelationReport,
    correlations: List[CorrelationResult]
) -> List[str]:
    """Generate human-readable insights from correlation data."""
    insights = []

    # Anomaly insights
    anomalies = [c for c in correlations if c.is_latency_spike]
    if anomalies:
        insights.append(
            f"Detected {len(anomalies)} latency spikes "
            f"({len(anomalies)/len(correlations)*100:.1f}% of tokens)"
        )

        # Group by cause
        causes = {}
        for a in anomalies:
            cause = a.spike_cause or "unknown"
            causes[cause] = causes.get(cause, 0) + 1

        for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
            insights.append(f"  - {cause}: {count} spikes")

    # Page fault correlation
    high_pf_tokens = [c for c in correlations if (c.page_faults or 0) > 0]
    if high_pf_tokens:
        pf_latencies = [c.app_latency_ms for c in high_pf_tokens]
        normal_latencies = [c.app_latency_ms for c in correlations if (c.page_faults or 0) == 0]
        if pf_latencies and normal_latencies:
            pf_mean = statistics.mean(pf_latencies)
            normal_mean = statistics.mean(normal_latencies)
            if pf_mean > normal_mean * 1.5:
                insights.append(
                    f"Page faults correlate with {pf_mean/normal_mean:.1f}x higher ITL "
                    f"(PF avg: {pf_mean:.2f}ms vs normal: {normal_mean:.2f}ms)"
                )

    # CUDA kernel correlation
    kernel_tokens = [c for c in correlations if c.cuda_kernel_duration_ms is not None]
    if kernel_tokens:
        insights.append(
            f"CUDA kernel data available for {len(kernel_tokens)}/{len(correlations)} tokens"
        )
        avg_kernel = statistics.mean(c.cuda_kernel_duration_ms for c in kernel_tokens)
        avg_itl = statistics.mean(c.app_latency_ms for c in kernel_tokens)
        kernel_pct = (avg_kernel / avg_itl * 100) if avg_itl > 0 else 0
        insights.append(
            f"GPU kernel time accounts for {kernel_pct:.1f}% of average ITL"
        )

    if not insights:
        insights.append("No anomalies detected. System is operating normally.")
        insights.append(
            "For deeper analysis, collect LTTng and Nsight traces simultaneously "
            "with benchmark execution."
        )

    return insights


def print_report(report: CorrelationReport):
    """Print formatted correlation report."""
    print(f"\n{'='*60}")
    print(f"PINSIGHT CROSS-LAYER CORRELATION REPORT")
    print(f"{'='*60}")
    print(f"Model:       {report.model}")
    print(f"Timestamp:   {report.timestamp}")
    print(f"Tokens:      {report.total_tokens}")

    print(f"\n{'─'*60}")
    print("APPLICATION LAYER:")
    print(f"  TTFT:              {report.ttft_ms:.2f} ms")
    print(f"  Avg ITL:           {report.avg_itl_ms:.2f} ms")
    print(f"  Tokens/s:          {report.tokens_per_second:.1f}")

    print(f"\n{'─'*60}")
    print("RUNTIME LAYER (CUDA):")
    if report.avg_kernel_duration_ms > 0:
        print(f"  Avg Kernel Time:   {report.avg_kernel_duration_ms:.2f} ms")
        print(f"  GPU Utilization:   {report.gpu_utilization_pct:.1f}%")
    else:
        print("  No CUDA trace data available.")
        print("  Collect with: nsys profile --trace=cuda python3 ...")

    print(f"\n{'─'*60}")
    print("SYSTEM LAYER:")
    print(f"  Page Faults:       {report.total_page_faults}")
    print(f"  Context Switches:  {report.total_context_switches}")

    print(f"\n{'─'*60}")
    print("ANOMALIES:")
    if report.anomalies:
        for i, a in enumerate(report.anomalies[:10], 1):
            print(f"  [{i}] Token {a['token_index']}: "
                  f"ITL={a['app_latency_ms']:.1f}ms "
                  f"(cause: {a.get('spike_cause', 'unknown')})")
        if len(report.anomalies) > 10:
            print(f"  ... and {len(report.anomalies) - 10} more")
    else:
        print("  None detected.")

    print(f"\n{'─'*60}")
    print("INSIGHTS:")
    for insight in report.insights:
        print(f"  • {insight}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="PInsight Trace Correlator - Cross-layer LLM inference analysis"
    )
    parser.add_argument(
        "--benchmark-results", type=str, required=True,
        help="Path to Ollama benchmark results JSON"
    )
    parser.add_argument(
        "--lttng-trace", type=str, default=None,
        help="Path to LTTng trace directory (CTF or pre-converted JSON)"
    )
    parser.add_argument(
        "--nsight-trace", type=str, default=None,
        help="Path to Nsight Systems JSON export"
    )
    parser.add_argument(
        "--model", type=str, default="unknown",
        help="Model name for report"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for correlation report"
    )
    parser.add_argument(
        "--window-ms", type=float, default=5.0,
        help="Correlation time window in milliseconds"
    )

    args = parser.parse_args()

    # Load benchmark results
    print(f"Loading benchmark results from {args.benchmark_results}...")
    benchmark = load_ollama_benchmark_results(args.benchmark_results)
    app_events = extract_token_events(benchmark)
    print(f"  Found {len(app_events)} token events")

    # Load runtime traces
    runtime_events = []
    if args.nsight_trace:
        print(f"Loading Nsight trace from {args.nsight_trace}...")
        runtime_events = load_nsight_trace(args.nsight_trace)
        print(f"  Found {len(runtime_events)} CUDA events")
    else:
        print("No Nsight trace provided (use --nsight-trace for GPU correlation)")

    # Load system traces
    system_events = []
    if args.lttng_trace:
        print(f"Loading LTTng trace from {args.lttng_trace}...")
        system_events = load_lttng_trace(args.lttng_trace)
        print(f"  Found {len(system_events)} system events")
    else:
        print("No LTTng trace provided (use --lttng-trace for system correlation)")

    # Correlate
    print("\nCorrelating traces...")
    window_ns = args.window_ms * 1e6
    correlations = correlate_traces(app_events, runtime_events, system_events, window_ns)

    # Generate report
    model = args.model or benchmark.get('model', 'unknown')
    report = generate_report(benchmark, correlations, model)

    # Print report
    print_report(report)

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Report saved to: {args.output}")


if __name__ == '__main__':
    main()
