#!/usr/bin/env python3
"""
PInsight vLLM KV Cache Profiler

Connects to a running vLLM server and collects KV cache metrics
for workload characterization and tuning feedback.

Metrics collected:
    - Per-request TTFT, TPOT, throughput
    - KV cache utilization over time (via /metrics endpoint)
    - Prefix cache hit rates
    - Active/waiting/swapped request counts
    - GPU memory usage

Usage:
    # Start vLLM server first, then:
    python vllm_kv_profiler.py --server http://localhost:8000 \\
        --workload workloads/dialogue.json --output results/baseline.json

    # Dry run (no server needed):
    python vllm_kv_profiler.py --dry-run --workload workloads/dialogue.json
"""

import json
import time
import argparse
import os
import sys
import statistics
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class RequestMetrics:
    """Metrics for a single inference request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0          # Time to first token
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    prompt_preview: str = ""


@dataclass
class ServerMetrics:
    """Metrics scraped from vLLM's /metrics endpoint."""
    timestamp: float = 0.0
    kv_cache_usage_pct: float = 0.0
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    num_requests_swapped: int = 0
    gpu_cache_blocks_used: int = 0
    gpu_cache_blocks_total: int = 0
    cpu_cache_blocks_used: int = 0
    prefix_cache_hit_rate: float = 0.0


@dataclass
class ProfilingReport:
    """Complete profiling report for a workload."""
    model: str
    workload_type: str
    server_url: str
    timestamp: str
    num_requests: int = 0
    # Request-level metrics
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    avg_tps: float = 0.0
    total_tokens: int = 0
    # Server-level metrics
    peak_kv_cache_usage: float = 0.0
    avg_kv_cache_usage: float = 0.0
    avg_prefix_hit_rate: float = 0.0
    peak_running_requests: int = 0
    # Raw data
    request_metrics: List[Dict] = field(default_factory=list)
    server_snapshots: List[Dict] = field(default_factory=list)


class VLLMProfiler:
    """Profiles a running vLLM server."""

    def __init__(self, server_url: str, model: str = None, dry_run: bool = False):
        self.server_url = server_url.rstrip('/')
        self.model = model
        self.dry_run = dry_run
        self._server_metrics: List[ServerMetrics] = []
        self._request_metrics: List[RequestMetrics] = []
        self._monitoring = False
        self._monitor_thread = None

    def check_server(self) -> bool:
        """Check if vLLM server is reachable."""
        if self.dry_run:
            print("[DRY RUN] Skipping server check")
            return True
        if not HAS_REQUESTS:
            print("ERROR: 'requests' package required. Install: pip install requests")
            return False
        try:
            r = requests.get(f"{self.server_url}/v1/models", timeout=5)
            if r.status_code == 200:
                data = r.json()
                models = data.get('data', [])
                if models and not self.model:
                    self.model = models[0].get('id', 'unknown')
                print(f"✓ vLLM server at {self.server_url}, model: {self.model}")
                return True
        except Exception as e:
            print(f"✗ Cannot reach vLLM server at {self.server_url}: {e}")
        return False

    def scrape_metrics(self) -> Optional[ServerMetrics]:
        """Scrape vLLM's Prometheus metrics endpoint."""
        if self.dry_run:
            return self._mock_server_metrics()
        try:
            r = requests.get(f"{self.server_url}/metrics", timeout=2)
            if r.status_code != 200:
                return None
            text = r.text
            sm = ServerMetrics(timestamp=time.time())
            for line in text.split('\n'):
                if line.startswith('#'):
                    continue
                if 'vllm:gpu_cache_usage_perc' in line:
                    sm.kv_cache_usage_pct = float(line.split()[-1]) * 100
                elif 'vllm:num_requests_running' in line:
                    sm.num_requests_running = int(float(line.split()[-1]))
                elif 'vllm:num_requests_waiting' in line:
                    sm.num_requests_waiting = int(float(line.split()[-1]))
                elif 'vllm:num_requests_swapped' in line:
                    sm.num_requests_swapped = int(float(line.split()[-1]))
            return sm
        except Exception:
            return None

    def _mock_server_metrics(self) -> ServerMetrics:
        """Generate mock metrics for dry-run mode."""
        import random
        return ServerMetrics(
            timestamp=time.time(),
            kv_cache_usage_pct=random.uniform(20, 80),
            num_requests_running=random.randint(1, 4),
            num_requests_waiting=random.randint(0, 2),
            gpu_cache_blocks_used=random.randint(100, 500),
            gpu_cache_blocks_total=1000,
            prefix_cache_hit_rate=random.uniform(0, 0.5),
        )

    def start_monitoring(self, interval_s: float = 0.5):
        """Start background thread to scrape server metrics."""
        self._monitoring = True
        def _monitor():
            while self._monitoring:
                m = self.scrape_metrics()
                if m:
                    self._server_metrics.append(m)
                time.sleep(interval_s)
        self._monitor_thread = threading.Thread(target=_monitor, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

    def send_request(self, prompt: str, max_tokens: int = 128) -> RequestMetrics:
        """Send a single inference request and measure metrics."""
        metrics = RequestMetrics(prompt_preview=prompt[:80])

        if self.dry_run:
            return self._mock_request(prompt, max_tokens)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }

        start = time.perf_counter()
        first_token_time = None
        token_count = 0

        try:
            r = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload, stream=True, timeout=120
            )
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get('choices', [{}])[0].get('delta', {})
                        if delta.get('content'):
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                    except json.JSONDecodeError:
                        continue

            end = time.perf_counter()
            metrics.total_time_ms = (end - start) * 1000
            metrics.completion_tokens = token_count
            if first_token_time:
                metrics.ttft_ms = (first_token_time - start) * 1000
            if metrics.total_time_ms > 0:
                metrics.tokens_per_second = token_count / ((end - start))
        except Exception as e:
            print(f"  Request failed: {e}")

        return metrics

    def _mock_request(self, prompt: str, max_tokens: int) -> RequestMetrics:
        """Generate mock request metrics for dry-run mode."""
        import random
        tokens = random.randint(50, max_tokens)
        ttft = random.uniform(20, 200)
        tps = random.uniform(30, 80)
        total = ttft + (tokens / tps * 1000)
        return RequestMetrics(
            prompt_tokens=len(prompt) // 4,
            completion_tokens=tokens,
            ttft_ms=ttft,
            total_time_ms=total,
            tokens_per_second=tps,
            prompt_preview=prompt[:80],
        )

    def run_workload(self, prompts: List[str], max_tokens: int = 128,
                     concurrency: int = 1) -> ProfilingReport:
        """Run a workload and collect profiling data."""
        print(f"\nProfiling {len(prompts)} requests "
              f"(max_tokens={max_tokens}, concurrency={concurrency})")

        self.start_monitoring()
        request_metrics = []

        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] Sending request...", end="", flush=True)
            m = self.send_request(prompt, max_tokens)
            request_metrics.append(m)
            print(f" TTFT={m.ttft_ms:.0f}ms, {m.tokens_per_second:.1f} tok/s, "
                  f"{m.completion_tokens} tokens")

        self.stop_monitoring()

        # Build report
        report = ProfilingReport(
            model=self.model or "unknown",
            workload_type="profiled",
            server_url=self.server_url,
            timestamp=datetime.now().isoformat(),
            num_requests=len(request_metrics),
        )

        if request_metrics:
            ttfts = [m.ttft_ms for m in request_metrics if m.ttft_ms > 0]
            tps_list = [m.tokens_per_second for m in request_metrics if m.tokens_per_second > 0]

            if ttfts:
                report.avg_ttft_ms = statistics.mean(ttfts)
                report.p50_ttft_ms = statistics.median(ttfts)
                report.p99_ttft_ms = sorted(ttfts)[int(len(ttfts) * 0.99)] if len(ttfts) > 1 else ttfts[0]
            if tps_list:
                report.avg_tps = statistics.mean(tps_list)
            report.total_tokens = sum(m.completion_tokens for m in request_metrics)
            report.request_metrics = [asdict(m) for m in request_metrics]

        if self._server_metrics:
            usages = [m.kv_cache_usage_pct for m in self._server_metrics]
            report.peak_kv_cache_usage = max(usages)
            report.avg_kv_cache_usage = statistics.mean(usages)
            report.peak_running_requests = max(
                m.num_requests_running for m in self._server_metrics)
            hit_rates = [m.prefix_cache_hit_rate for m in self._server_metrics
                         if m.prefix_cache_hit_rate > 0]
            if hit_rates:
                report.avg_prefix_hit_rate = statistics.mean(hit_rates)
            report.server_snapshots = [asdict(m) for m in self._server_metrics]

        return report


def load_prompts(filepath: str) -> List[str]:
    """Load prompts from a JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, list):
        return [p if isinstance(p, str) else p.get('prompt', p.get('content', str(p)))
                for p in data]
    elif isinstance(data, dict):
        return data.get('prompts', [data.get('prompt', '')])
    return []


def print_report(report: ProfilingReport):
    """Pretty-print profiling report."""
    print(f"\n{'='*60}")
    print(f"PINSIGHT vLLM PROFILING REPORT")
    print(f"{'='*60}")
    print(f"Model:    {report.model}")
    print(f"Server:   {report.server_url}")
    print(f"Requests: {report.num_requests}")
    print(f"Tokens:   {report.total_tokens}")

    print(f"\n{'─'*60}")
    print("REQUEST METRICS:")
    print(f"  Avg TTFT:    {report.avg_ttft_ms:.1f} ms")
    print(f"  P50 TTFT:    {report.p50_ttft_ms:.1f} ms")
    print(f"  P99 TTFT:    {report.p99_ttft_ms:.1f} ms")
    print(f"  Avg TPS:     {report.avg_tps:.1f} tok/s")

    print(f"\n{'─'*60}")
    print("KV CACHE:")
    print(f"  Peak Usage:  {report.peak_kv_cache_usage:.1f}%")
    print(f"  Avg Usage:   {report.avg_kv_cache_usage:.1f}%")
    if report.avg_prefix_hit_rate > 0:
        print(f"  Prefix Hits: {report.avg_prefix_hit_rate:.1%}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="PInsight vLLM KV Cache Profiler")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                        help="vLLM server URL")
    parser.add_argument("--workload", type=str, help="JSON file with prompts")
    parser.add_argument("--prompt", type=str, help="Single prompt to profile")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with mock data (no server needed)")
    args = parser.parse_args()

    profiler = VLLMProfiler(args.server, model=args.model, dry_run=args.dry_run)

    if not profiler.check_server():
        if not args.dry_run:
            print("\nHint: Start vLLM first, or use --dry-run for testing")
            sys.exit(1)

    if args.workload:
        prompts = load_prompts(args.workload)
    elif args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "Explain the concept of attention in transformer models.",
            "What are the trade-offs between token eviction and chunking for KV cache?",
            "Write a Python function that implements a sliding window attention mechanism.",
        ]

    report = profiler.run_workload(prompts, args.max_tokens)
    print_report(report)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Report saved to: {args.output}")


if __name__ == '__main__':
    main()
