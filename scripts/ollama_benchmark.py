#!/usr/bin/env python3
"""
Ollama LLM Inference Benchmark
Tracks: Time to First Token (TTFT), Time Between Tokens (TBT),
        Time Per Output Token (TPOT), Inter-Token Latency (ITL)
"""

import time
import json
import statistics
import argparse
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import ollama


@dataclass
class TokenTiming:
    """Timing data for a single token."""
    token_index: int
    token: str
    timestamp: float  # absolute timestamp
    latency: float    # time since last token (or start for first token)


@dataclass
class InferenceMetrics:
    """Comprehensive metrics for a single inference request."""
    prompt: str
    model: str

    # Timing metrics
    request_start: float = 0.0
    first_token_time: float = 0.0
    last_token_time: float = 0.0
    request_end: float = 0.0

    # Token data
    token_timings: List[TokenTiming] = field(default_factory=list)
    total_tokens: int = 0
    output_text: str = ""

    # Derived metrics (computed after collection)
    ttft: float = 0.0              # Time to First Token
    total_generation_time: float = 0.0
    tpot: float = 0.0              # Time Per Output Token (average)
    itl_mean: float = 0.0          # Inter-Token Latency (mean)
    itl_std: float = 0.0           # Inter-Token Latency (std dev)
    itl_min: float = 0.0           # Inter-Token Latency (min)
    itl_max: float = 0.0           # Inter-Token Latency (max)
    itl_p50: float = 0.0           # Inter-Token Latency (median)
    itl_p90: float = 0.0           # Inter-Token Latency (90th percentile)
    itl_p99: float = 0.0           # Inter-Token Latency (99th percentile)
    tokens_per_second: float = 0.0


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_inference_with_timing(
    client: ollama.Client,
    model: str,
    prompt: str,
    verbose: bool = False,
    token_log_file=None
) -> InferenceMetrics:
    """
    Run inference and collect detailed timing metrics.
    Uses streaming to capture per-token timing.

    Args:
        client: Ollama client
        model: Model name
        prompt: Input prompt
        verbose: Print per-token timing to terminal
        token_log_file: File handle to write token timing log
    """
    metrics = InferenceMetrics(prompt=prompt, model=model)

    # Start timing
    metrics.request_start = time.perf_counter()
    last_token_time = metrics.request_start

    # Stream response to capture per-token timing
    stream = client.generate(
        model=model,
        prompt=prompt,
        stream=True
    )

    token_index = 0
    for chunk in stream:
        current_time = time.perf_counter()
        token = chunk.get('response', '')

        if token:
            latency = current_time - last_token_time

            timing = TokenTiming(
                token_index=token_index,
                token=token,
                timestamp=current_time,
                latency=latency
            )
            metrics.token_timings.append(timing)

            if token_index == 0:
                metrics.first_token_time = current_time

            metrics.output_text += token
            last_token_time = current_time
            token_index += 1

            # Format token output line
            token_line = f"Token {token_index}: '{token}' | Latency: {latency*1000:.2f}ms"

            if verbose:
                print(token_line)

            if token_log_file:
                token_log_file.write(token_line + "\n")

    metrics.last_token_time = last_token_time
    metrics.request_end = time.perf_counter()
    metrics.total_tokens = token_index

    # Compute derived metrics
    compute_derived_metrics(metrics)

    return metrics


def compute_derived_metrics(metrics: InferenceMetrics) -> None:
    """Compute derived metrics from raw timing data."""
    if metrics.total_tokens == 0:
        return

    # Time to First Token
    metrics.ttft = metrics.first_token_time - metrics.request_start

    # Total generation time
    metrics.total_generation_time = metrics.request_end - metrics.request_start

    # Time Per Output Token (average)
    if metrics.total_tokens > 0:
        metrics.tpot = metrics.total_generation_time / metrics.total_tokens

    # Inter-Token Latencies (skip first token for ITL calculation)
    if len(metrics.token_timings) > 1:
        itl_values = [t.latency for t in metrics.token_timings[1:]]

        metrics.itl_mean = statistics.mean(itl_values)
        metrics.itl_std = statistics.stdev(itl_values) if len(itl_values) > 1 else 0.0
        metrics.itl_min = min(itl_values)
        metrics.itl_max = max(itl_values)
        metrics.itl_p50 = percentile(itl_values, 50)
        metrics.itl_p90 = percentile(itl_values, 90)
        metrics.itl_p99 = percentile(itl_values, 99)

    # Tokens per second
    if metrics.total_generation_time > 0:
        metrics.tokens_per_second = metrics.total_tokens / metrics.total_generation_time


def print_metrics_summary(metrics: InferenceMetrics) -> None:
    """Print a formatted summary of metrics."""
    print("\n" + "="*60)
    print("OLLAMA INFERENCE METRICS")
    print("="*60)
    print(f"Model: {metrics.model}")
    print(f"Prompt length: {len(metrics.prompt)} chars")
    print(f"Output tokens: {metrics.total_tokens}")
    print("-"*60)
    print("TIMING METRICS:")
    print(f"  Time to First Token (TTFT):     {metrics.ttft*1000:10.2f} ms")
    print(f"  Total Generation Time:          {metrics.total_generation_time*1000:10.2f} ms")
    print(f"  Time Per Output Token (TPOT):   {metrics.tpot*1000:10.2f} ms")
    print(f"  Tokens Per Second:              {metrics.tokens_per_second:10.2f} tok/s")
    print("-"*60)
    print("INTER-TOKEN LATENCY (ITL):")
    print(f"  Mean:                           {metrics.itl_mean*1000:10.2f} ms")
    print(f"  Std Dev:                        {metrics.itl_std*1000:10.2f} ms")
    print(f"  Min:                            {metrics.itl_min*1000:10.2f} ms")
    print(f"  Max:                            {metrics.itl_max*1000:10.2f} ms")
    print(f"  P50 (Median):                   {metrics.itl_p50*1000:10.2f} ms")
    print(f"  P90:                            {metrics.itl_p90*1000:10.2f} ms")
    print(f"  P99:                            {metrics.itl_p99*1000:10.2f} ms")
    print("="*60)


def run_benchmark(
    host: str,
    model: str,
    prompts: List[str],
    output_file: Optional[str] = None,
    token_log: Optional[str] = None,
    verbose: bool = False
) -> List[InferenceMetrics]:
    """
    Run benchmark with multiple prompts and collect metrics.

    Args:
        host: Ollama server URL
        model: Model name
        prompts: List of prompts to test
        output_file: JSON file for results
        token_log: Text file for per-token timing log
        verbose: Print per-token timing to terminal
    """
    client = ollama.Client(host=host)

    print(f"Connecting to Ollama server at {host}")
    print(f"Model: {model}")
    print(f"Running {len(prompts)} inference requests...")
    if token_log:
        print(f"Token log: {token_log}")
    print()

    all_metrics = []

    # Open token log file if specified
    token_log_file = open(token_log, 'w') if token_log else None

    try:
        for i, prompt in enumerate(prompts):
            print(f"Request {i+1}/{len(prompts)}: {prompt[:50]}...")

            # Write request header to token log
            if token_log_file:
                token_log_file.write(f"\n{'='*60}\n")
                token_log_file.write(f"Request {i+1}/{len(prompts)}\n")
                token_log_file.write(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n")
                token_log_file.write(f"{'='*60}\n")

            try:
                metrics = run_inference_with_timing(
                    client, model, prompt, verbose, token_log_file
                )
                all_metrics.append(metrics)
                print_metrics_summary(metrics)

                # Write summary to token log
                if token_log_file:
                    token_log_file.write(f"\n--- Summary ---\n")
                    token_log_file.write(f"Total tokens: {metrics.total_tokens}\n")
                    token_log_file.write(f"TTFT: {metrics.ttft*1000:.2f}ms\n")
                    token_log_file.write(f"TPOT: {metrics.tpot*1000:.2f}ms\n")
                    token_log_file.write(f"ITL Mean: {metrics.itl_mean*1000:.2f}ms\n")
                    token_log_file.write(f"Throughput: {metrics.tokens_per_second:.2f} tok/s\n")

            except Exception as e:
                print(f"Error: {e}")
                if token_log_file:
                    token_log_file.write(f"ERROR: {e}\n")
                continue
    finally:
        if token_log_file:
            token_log_file.close()

    # Aggregate summary
    if len(all_metrics) > 1:
        print("\n" + "="*60)
        print("AGGREGATE SUMMARY (all requests)")
        print("="*60)

        ttfts = [m.ttft for m in all_metrics]
        tpots = [m.tpot for m in all_metrics]
        itl_means = [m.itl_mean for m in all_metrics]
        tps_values = [m.tokens_per_second for m in all_metrics]

        print(f"Requests: {len(all_metrics)}")
        print(f"TTFT:  Mean={statistics.mean(ttfts)*1000:.2f}ms, Std={statistics.stdev(ttfts)*1000:.2f}ms")
        print(f"TPOT:  Mean={statistics.mean(tpots)*1000:.2f}ms, Std={statistics.stdev(tpots)*1000:.2f}ms")
        print(f"ITL:   Mean={statistics.mean(itl_means)*1000:.2f}ms")
        print(f"TPS:   Mean={statistics.mean(tps_values):.2f} tok/s")
        print("="*60)

    # Save results
    if output_file:
        results = {
            "timestamp": datetime.now().isoformat(),
            "host": host,
            "model": model,
            "metrics": []
        }
        for m in all_metrics:
            m_dict = asdict(m)
            # Convert token_timings to serializable format
            m_dict['token_timings'] = [asdict(t) for t in m.token_timings]
            results["metrics"].append(m_dict)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    if token_log:
        print(f"Token log saved to: {token_log}")

    return all_metrics


# Default test prompts
DEFAULT_PROMPTS = [
    "Explain the concept of recursion in programming in 3 sentences.",
    "What are the main differences between Python and C++?",
    "Write a short poem about artificial intelligence.",
    "Describe the Big O notation and why it matters.",
    "What is the purpose of a hash table data structure?",
]


def main():
    parser = argparse.ArgumentParser(description='Ollama LLM Inference Benchmark')
    parser.add_argument('--host', default='http://localhost:11434',
                        help='Ollama server URL')
    parser.add_argument('--model', default='llama2',
                        help='Model name to benchmark')
    parser.add_argument('--prompt', type=str,
                        help='Single prompt to test (overrides default prompts)')
    parser.add_argument('--prompts-file', type=str,
                        help='JSON file with list of prompts')
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file for results')
    parser.add_argument('--token-log', '-t', type=str,
                        help='Output text file for per-token timing log')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print per-token timing to terminal')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of times to run each prompt')

    args = parser.parse_args()

    # Determine prompts
    if args.prompt:
        prompts = [args.prompt] * args.num_runs
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
        prompts = prompts * args.num_runs
    else:
        prompts = DEFAULT_PROMPTS * args.num_runs

    # Run benchmark
    run_benchmark(
        host=args.host,
        model=args.model,
        prompts=prompts,
        output_file=args.output,
        token_log=args.token_log,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
