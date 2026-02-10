#!/usr/bin/env python3
"""
Reasoning Benchmark Script
Evaluates LLM reasoning capabilities using complex HPC-domain tasks.
Measures both response quality (scoring) and inference performance (timing).

Integrates with PInsight project context:
- MPI communication analysis
- OpenMP task scheduling
- CUDA kernel optimization
- Cross-layer trace correlation
- Performance anomaly detection
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

try:
    import ollama
except ImportError:
    print("Error: ollama package required. Install with: pip install ollama")
    sys.exit(1)


@dataclass
class ReasoningResult:
    """Result from a single reasoning task evaluation."""
    task_id: str
    category: str
    difficulty: str
    prompt: str
    response: str
    # Timing metrics
    ttft: float = 0.0
    total_time: float = 0.0
    total_tokens: int = 0
    tpot: float = 0.0
    tokens_per_second: float = 0.0
    # Quality metrics
    keyword_matches: List[str] = field(default_factory=list)
    keyword_score: float = 0.0
    rubric_scores: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    max_score: float = 0.0
    score_percentage: float = 0.0
    # Response analysis
    response_length: int = 0
    reasoning_depth: int = 0  # Number of distinct reasoning steps


@dataclass
class BenchmarkSummary:
    """Summary of the full reasoning benchmark run."""
    model: str
    timestamp: str
    total_tasks: int = 0
    completed_tasks: int = 0
    # Aggregate timing
    avg_ttft: float = 0.0
    avg_tpot: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_total_time: float = 0.0
    # Aggregate quality
    avg_score_percentage: float = 0.0
    category_scores: Dict[str, float] = field(default_factory=dict)
    difficulty_scores: Dict[str, float] = field(default_factory=dict)
    # Individual results
    results: List[Dict] = field(default_factory=list)


def load_tasks(tasks_file: str) -> List[Dict]:
    """Load reasoning tasks from JSON file."""
    with open(tasks_file, 'r') as f:
        data = json.load(f)
    return data.get('tasks', [])


def run_reasoning_task(
    client: ollama.Client,
    model: str,
    task: Dict,
    verbose: bool = False
) -> ReasoningResult:
    """
    Run a single reasoning task and collect timing + quality metrics.

    Uses streaming to capture per-token timing, similar to ollama_benchmark.py.
    """
    prompt = task['prompt']
    result = ReasoningResult(
        task_id=task['id'],
        category=task['category'],
        difficulty=task['difficulty'],
        prompt=prompt,
        response="",
        max_score=task.get('max_score', 10)
    )

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Task: {task['id']} ({task['category']}, {task['difficulty']})")
        print(f"Prompt: {prompt[:100]}...")
        print(f"{'─'*60}")

    # System prompt for reasoning tasks
    system_prompt = (
        "You are an expert in high-performance computing, parallel programming, "
        "GPU computing, and LLM inference optimization. Provide detailed, "
        "technically accurate analysis with specific recommendations. "
        "Show your reasoning step by step."
    )

    request_start = time.perf_counter()
    first_token_time = None
    tokens = []
    token_times = []

    try:
        stream = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        for chunk in stream:
            current_time = time.perf_counter()

            token = chunk.get('message', {}).get('content', '')
            if token:
                if first_token_time is None:
                    first_token_time = current_time
                tokens.append(token)
                token_times.append(current_time)

                if verbose:
                    print(token, end='', flush=True)

    except Exception as e:
        print(f"\nError running task {task['id']}: {e}")
        result.response = f"ERROR: {str(e)}"
        return result

    request_end = time.perf_counter()

    # Compute timing metrics
    result.response = ''.join(tokens)
    result.total_tokens = len(tokens)
    result.response_length = len(result.response)

    if first_token_time is not None:
        result.ttft = first_token_time - request_start

    result.total_time = request_end - request_start

    if result.total_tokens > 0:
        result.tpot = result.total_time / result.total_tokens
        result.tokens_per_second = result.total_tokens / result.total_time

    # Score the response
    score_response(result, task)

    if verbose:
        print(f"\n\n{'─'*40}")
        print(f"TTFT: {result.ttft*1000:.1f}ms | "
              f"Tokens: {result.total_tokens} | "
              f"Speed: {result.tokens_per_second:.1f} tok/s | "
              f"Score: {result.total_score}/{result.max_score} "
              f"({result.score_percentage:.0f}%)")
        print(f"Keywords matched: {', '.join(result.keyword_matches)}")

    return result


def score_response(result: ReasoningResult, task: Dict):
    """
    Score a reasoning response based on expected keywords and rubric.

    Scoring approach:
    1. Keyword matching (automatic) - checks for domain-specific terms
    2. Reasoning depth analysis (automatic) - counts distinct analysis steps
    3. Rubric-based scoring (heuristic) - checks for structural elements
    """
    response_lower = result.response.lower()

    # 1. Keyword matching
    expected_keywords = task.get('expected_keywords', [])
    matched = []
    for keyword in expected_keywords:
        if keyword.lower() in response_lower:
            matched.append(keyword)
    result.keyword_matches = matched

    # Keyword score: proportion of expected keywords found
    if expected_keywords:
        result.keyword_score = len(matched) / len(expected_keywords)
    else:
        result.keyword_score = 0.0

    # 2. Reasoning depth: count numbered lists, bullet points, headers
    reasoning_indicators = [
        len(re.findall(r'^\d+[\.\)]\s', result.response, re.MULTILINE)),  # Numbered steps
        len(re.findall(r'^[-*•]\s', result.response, re.MULTILINE)),      # Bullet points
        len(re.findall(r'^#{1,4}\s', result.response, re.MULTILINE)),     # Headers
        len(re.findall(r'(first|second|third|finally|moreover|additionally|furthermore)',
                       response_lower)),  # Transition words
    ]
    result.reasoning_depth = sum(reasoning_indicators)

    # 3. Rubric-based scoring (heuristic auto-scoring)
    rubric = task.get('scoring_rubric', {})
    total_score = 0.0

    for criterion, max_points in rubric.items():
        # Heuristic scoring: award points based on keyword density
        # and response quality indicators
        criterion_keywords = criterion.lower().replace('_', ' ').split()
        criterion_matches = sum(
            1 for kw in criterion_keywords
            if kw in response_lower and len(kw) > 3
        )

        # Score based on keyword match ratio and response depth
        if criterion_matches > 0 and result.reasoning_depth > 3:
            score = max_points * min(1.0, (result.keyword_score * 0.6 + 0.4))
        elif criterion_matches > 0:
            score = max_points * 0.5
        elif result.keyword_score > 0.3:
            score = max_points * 0.3
        else:
            score = 0.0

        result.rubric_scores[criterion] = round(score, 1)
        total_score += score

    result.total_score = round(total_score, 1)
    result.score_percentage = (total_score / result.max_score * 100) if result.max_score > 0 else 0


def run_benchmark(
    host: str,
    model: str,
    tasks_file: str,
    categories: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    output_file: Optional[str] = None,
    verbose: bool = False
) -> BenchmarkSummary:
    """
    Run the full reasoning benchmark.

    Args:
        host: Ollama server URL
        model: Model name
        tasks_file: Path to reasoning tasks JSON
        categories: Filter by categories (None = all)
        difficulties: Filter by difficulty (None = all)
        max_tasks: Maximum number of tasks to run
        output_file: Output JSON file path
        verbose: Enable verbose output
    """
    client = ollama.Client(host=host)

    # Load and filter tasks
    all_tasks = load_tasks(tasks_file)

    if categories:
        all_tasks = [t for t in all_tasks if t['category'] in categories]
    if difficulties:
        all_tasks = [t for t in all_tasks if t['difficulty'] in difficulties]
    if max_tasks:
        all_tasks = all_tasks[:max_tasks]

    print(f"\n{'='*60}")
    print(f"REASONING BENCHMARK")
    print(f"{'='*60}")
    print(f"Model:      {model}")
    print(f"Tasks:      {len(all_tasks)}")
    print(f"Categories: {', '.join(set(t['category'] for t in all_tasks))}")
    print(f"{'='*60}\n")

    summary = BenchmarkSummary(
        model=model,
        timestamp=datetime.now().isoformat(),
        total_tasks=len(all_tasks),
    )

    results = []
    for i, task in enumerate(all_tasks, 1):
        print(f"[{i}/{len(all_tasks)}] Running: {task['id']}...", end=" ", flush=True)

        result = run_reasoning_task(client, model, task, verbose=verbose)
        results.append(result)

        if not verbose:
            print(f"TTFT={result.ttft*1000:.0f}ms | "
                  f"{result.tokens_per_second:.1f} tok/s | "
                  f"Score: {result.score_percentage:.0f}%")

    # Compute aggregates
    summary.completed_tasks = len(results)
    summary.results = [asdict(r) for r in results]

    if results:
        summary.avg_ttft = statistics.mean(r.ttft for r in results)
        summary.avg_tpot = statistics.mean(r.tpot for r in results if r.tpot > 0)
        summary.avg_tokens_per_second = statistics.mean(
            r.tokens_per_second for r in results if r.tokens_per_second > 0
        )
        summary.avg_total_time = statistics.mean(r.total_time for r in results)
        summary.avg_score_percentage = statistics.mean(
            r.score_percentage for r in results
        )

        # Category scores
        cat_scores = {}
        for r in results:
            if r.category not in cat_scores:
                cat_scores[r.category] = []
            cat_scores[r.category].append(r.score_percentage)
        summary.category_scores = {
            cat: statistics.mean(scores) for cat, scores in cat_scores.items()
        }

        # Difficulty scores
        diff_scores = {}
        for r in results:
            if r.difficulty not in diff_scores:
                diff_scores[r.difficulty] = []
            diff_scores[r.difficulty].append(r.score_percentage)
        summary.difficulty_scores = {
            diff: statistics.mean(scores) for diff, scores in diff_scores.items()
        }

    # Print summary
    print_benchmark_summary(summary)

    # Save results
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return summary


def print_benchmark_summary(summary: BenchmarkSummary):
    """Print formatted benchmark summary."""
    print(f"\n{'='*60}")
    print(f"REASONING BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Model:           {summary.model}")
    print(f"Tasks Completed: {summary.completed_tasks}/{summary.total_tasks}")

    print(f"\n{'─'*60}")
    print("PERFORMANCE METRICS:")
    print(f"  Avg TTFT:        {summary.avg_ttft*1000:.1f} ms")
    print(f"  Avg TPOT:        {summary.avg_tpot*1000:.1f} ms")
    print(f"  Avg Tokens/s:    {summary.avg_tokens_per_second:.1f}")
    print(f"  Avg Total Time:  {summary.avg_total_time:.1f} s")

    print(f"\n{'─'*60}")
    print("QUALITY SCORES:")
    print(f"  Overall:         {summary.avg_score_percentage:.1f}%")

    if summary.category_scores:
        print(f"\n  By Category:")
        for cat, score in sorted(summary.category_scores.items()):
            bar = '█' * int(score / 5) + '░' * (20 - int(score / 5))
            print(f"    {cat:25s} {bar} {score:.1f}%")

    if summary.difficulty_scores:
        print(f"\n  By Difficulty:")
        for diff, score in sorted(summary.difficulty_scores.items()):
            bar = '█' * int(score / 5) + '░' * (20 - int(score / 5))
            print(f"    {diff:25s} {bar} {score:.1f}%")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reasoning Benchmark for LLM Inference Analysis"
    )
    parser.add_argument(
        "--host", type=str, default="http://localhost:11434",
        help="Ollama server URL"
    )
    parser.add_argument(
        "--model", type=str, default="llama2",
        help="Model to benchmark"
    )
    parser.add_argument(
        "--tasks", type=str, default="prompts/reasoning_tasks.json",
        help="Path to reasoning tasks JSON file"
    )
    parser.add_argument(
        "--categories", type=str, nargs="*", default=None,
        choices=["mpi_analysis", "openmp_analysis", "cuda_optimization",
                 "cross_layer", "anomaly_detection", "general_reasoning"],
        help="Filter by task categories"
    )
    parser.add_argument(
        "--difficulty", type=str, nargs="*", default=None,
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty"
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None,
        help="Maximum number of tasks to run"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show full responses and per-token timing"
    )

    args = parser.parse_args()

    if not os.path.exists(args.tasks):
        print(f"Error: Tasks file not found: {args.tasks}")
        print("Run from the project root directory.")
        sys.exit(1)

    run_benchmark(
        host=args.host,
        model=args.model,
        tasks_file=args.tasks,
        categories=args.categories,
        difficulties=args.difficulty,
        max_tasks=args.max_tasks,
        output_file=args.output,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
