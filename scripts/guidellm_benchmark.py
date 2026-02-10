#!/usr/bin/env python3
"""
GuideLLM Benchmark Wrapper
Runs GuideLLM benchmarks against OpenAI-compatible endpoints (Ollama, vLLM, etc.)
and normalizes results for comparison with Ollama and Etalon benchmarks.

Supports:
- Text-only workloads (chat completions, text completions)
- Multimodal workloads (image+text via chat completions)
- Multiple load profiles (synchronous, concurrent, sweep, constant, poisson)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def check_guidellm_installed() -> bool:
    """Check if GuideLLM is installed."""
    try:
        result = subprocess.run(
            ["guidellm", "--version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_guidellm():
    """Install GuideLLM with recommended dependencies."""
    print("Installing GuideLLM...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "guidellm[recommended]"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Failed to install GuideLLM: {result.stderr}")
        sys.exit(1)
    print("GuideLLM installed successfully.")


def run_guidellm_benchmark(
    target: str,
    model: str,
    profile: str = "sweep",
    rate: Optional[int] = None,
    max_seconds: int = 30,
    max_requests: Optional[int] = None,
    data: str = "prompt_tokens=256,output_tokens=128",
    request_type: str = "chat_completions",
    processor: Optional[str] = None,
    output_dir: str = "./results/guidellm",
    verbose: bool = False,
) -> Dict:
    """
    Run a GuideLLM benchmark.

    Args:
        target: OpenAI-compatible API endpoint URL
        model: Model name/identifier
        profile: Load profile (synchronous, concurrent, throughput, constant, poisson, sweep)
        rate: Rate value (meaning depends on profile)
        max_seconds: Maximum duration per benchmark
        max_requests: Maximum requests per benchmark (alternative to max_seconds)
        data: Data specification (synthetic, HuggingFace dataset, or file path)
        request_type: API request type (chat_completions, completions, audio_transcription)
        processor: HuggingFace tokenizer/processor ID (optional)
        output_dir: Output directory for results
        verbose: Enable verbose output

    Returns:
        Dict with benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(output_dir, f"guidellm_{timestamp}")

    # Fix for Ollama model names (e.g. 'llama2' -> 'llama2:latest')
    # GuideLLM requires exact match with what the API returns.
    if ":" not in model:
        print(f"Warning: Model '{model}' has no tag. Appending ':latest' for GuideLLM compatibility.")
        model = f"{model}:latest"
        
    # Auto-detect processor/tokenizer if not provided, to avoid using gated models (like meta-llama/Llama-2-7b-chat-hf)
    if not processor:
        model_lower = model.lower()
        if "llama" in model_lower:
             # Use NousResearch mirror (non-gated)
             processor = "NousResearch/Llama-2-7b-chat-hf"
             print(f"Using default non-gated processor for Llama: {processor}")
        elif "mistral" in model_lower:
             processor = "mistralai/Mistral-7B-Instruct-v0.2"
             print(f"Using default processor for Mistral: {processor}")

    # Build command with new CLI arguments
    # Note: --profile is now --rate-type, --output-dir is --output-path
    cmd = [
        "guidellm", "benchmark", "run",
        "--target", target,
        "--rate-type", profile,
        "--data", data,
        "--output-path", output_dir,
        "--model", model,
    ]
    
    if processor:
        cmd.extend(["--processor", processor])

    if rate is not None:
        cmd.extend(["--rate", str(rate)])
    
    if max_seconds:
        cmd.extend(["--max-seconds", str(max_seconds)])
        
    if max_requests:
        cmd.extend(["--max-requests", str(max_requests)])

    print(f"\n{'='*60}")
    print(f"GUIDELLM BENCHMARK")
    print(f"{'='*60}")
    print(f"Target:       {target}")
    print(f"Model:        {model}")
    print(f"Processor:    {processor or 'Auto'}")
    print(f"Profile:      {profile}")
    print(f"Data:         {data}")
    print(f"Output Path:  {output_dir}")
    print(f"Command:      {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start_time = time.time()
    elapsed = 0

    try:
        process = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=max_seconds * 10 if not max_requests else 600
        )

        elapsed = time.time() - start_time

        if process.returncode != 0:
            print(f"GuideLLM benchmark failed (exit code {process.returncode})")
            if not verbose and process.stderr:
                print(f"stderr: {process.stderr[:1000]}")
            return {"error": "benchmark_failed", "exit_code": process.returncode}

    except subprocess.TimeoutExpired:
        print("GuideLLM benchmark timed out")
        return {"error": "timeout"}

    results = load_guidellm_results(output_dir)
    results["benchmark_metadata"] = {
        "tool": "guidellm",
        "target": target,
        "model": model,
        "processor": processor,
        "profile": profile,
        "data": data,
        "request_type": request_type,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    # Save normalized results
    normalized_path = os.path.join(output_dir, "normalized_results.json")
    normalized = normalize_results(results)
    with open(normalized_path, "w") as f:
        json.dump(normalized, f, indent=2)
    print(f"\nNormalized results saved to: {normalized_path}")

    return normalized


def load_guidellm_results(output_dir: str) -> Dict:
    """Load results from GuideLLM output directory."""
    results = {}

    # Look for benchmarks.json (primary output)
    json_path = os.path.join(output_dir, "benchmarks.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            results["raw"] = json.load(f)
        print(f"Loaded GuideLLM results from {json_path}")
    else:
        # Search for any JSON files
        json_files = list(Path(output_dir).glob("*.json"))
        if json_files:
            # Filter out normalized_results.json to avoid self-loading
            json_files = [p for p in json_files if p.name != "normalized_results.json"]
            
        if json_files:
            with open(json_files[0], "r") as f:
                results["raw"] = json.load(f)
            print(f"Loaded GuideLLM results from {json_files[0]}")
        else:
            print(f"No GuideLLM results found in {output_dir}")
            results["raw"] = {}

    return results


def normalize_results(results: Dict) -> Dict:
    """
    Normalize GuideLLM results to a common format for comparison
    with Ollama and Etalon benchmarks.

    Output format:
    {
        "tool": "guidellm",
        "metrics": {
            "ttft": { "mean", "p50", "p90", "p99", "min", "max" },
            "itl": { "mean", "p50", "p90", "p99", "min", "max" },
            "tpot": { "mean" },
            "throughput": { "tokens_per_second", "requests_per_second" },
            "latency": { "mean", "p50", "p90", "p99" }
        },
        "benchmark_metadata": { ... },
        "raw": { ... }
    }
    """
    normalized = {
        "tool": "guidellm",
        "metrics": {
            "ttft": {},
            "itl": {},
            "tpot": {},
            "throughput": {},
            "latency": {},
        },
        "benchmark_metadata": results.get("benchmark_metadata", {}),
    }

    raw = results.get("raw", {})

    # GuideLLM output structure varies by version. Attempt to extract
    # metrics from the known structure.
    if isinstance(raw, dict):
        # Try to extract from benchmarks array
        benchmarks = raw.get("benchmarks", [])
        if benchmarks and isinstance(benchmarks, list):
            # Use the last benchmark (highest load in sweep) or first available
            bench = benchmarks[-1] if len(benchmarks) > 1 else benchmarks[0]

            stats = bench.get("stats", bench.get("statistics", {}))

            # TTFT
            ttft_data = stats.get("ttft", stats.get("time_to_first_token", {}))
            if isinstance(ttft_data, dict):
                normalized["metrics"]["ttft"] = {
                    "mean": ttft_data.get("mean", ttft_data.get("avg")),
                    "p50": ttft_data.get("p50", ttft_data.get("median")),
                    "p90": ttft_data.get("p90"),
                    "p99": ttft_data.get("p99"),
                    "min": ttft_data.get("min"),
                    "max": ttft_data.get("max"),
                }

            # ITL
            itl_data = stats.get("itl", stats.get("inter_token_latency", {}))
            if isinstance(itl_data, dict):
                normalized["metrics"]["itl"] = {
                    "mean": itl_data.get("mean", itl_data.get("avg")),
                    "p50": itl_data.get("p50", itl_data.get("median")),
                    "p90": itl_data.get("p90"),
                    "p99": itl_data.get("p99"),
                    "min": itl_data.get("min"),
                    "max": itl_data.get("max"),
                }

            # Throughput
            throughput_data = stats.get("throughput", {})
            if isinstance(throughput_data, dict):
                normalized["metrics"]["throughput"] = {
                    "tokens_per_second": throughput_data.get(
                        "output_tokens_per_second",
                        throughput_data.get("tokens_per_second")
                    ),
                    "requests_per_second": throughput_data.get(
                        "requests_per_second",
                        throughput_data.get("rps")
                    ),
                }

            # End-to-end latency
            latency_data = stats.get("request_latency", stats.get("latency", {}))
            if isinstance(latency_data, dict):
                normalized["metrics"]["latency"] = {
                    "mean": latency_data.get("mean", latency_data.get("avg")),
                    "p50": latency_data.get("p50", latency_data.get("median")),
                    "p90": latency_data.get("p90"),
                    "p99": latency_data.get("p99"),
                }

    # Keep raw data for detailed analysis
    normalized["raw"] = raw

    return normalized


def print_summary(results: Dict):
    """Print human-readable summary of GuideLLM results."""
    metrics = results.get("metrics", {})

    print(f"\n{'='*60}")
    print(f"GUIDELLM RESULTS SUMMARY")
    print(f"{'='*60}")

    meta = results.get("benchmark_metadata", {})
    if meta:
        print(f"Model:    {meta.get('model', 'N/A')}")
        print(f"Profile:  {meta.get('profile', 'N/A')}")
        print(f"Processor:{meta.get('processor', 'N/A')}")
        print(f"Duration: {meta.get('elapsed_seconds', 0):.1f}s")

    print(f"\n{'─'*60}")
    print("TIMING METRICS:")

    ttft = metrics.get("ttft", {})
    if ttft.get("mean") is not None:
        print(f"  TTFT Mean:    {ttft['mean']*1000:.2f} ms") if ttft["mean"] < 10 else print(f"  TTFT Mean:    {ttft['mean']:.2f} ms")
        if ttft.get("p50"):
            print(f"  TTFT P50:     {ttft['p50']*1000:.2f} ms") if ttft["p50"] < 10 else print(f"  TTFT P50:     {ttft['p50']:.2f} ms")
        if ttft.get("p99"):
            print(f"  TTFT P99:     {ttft['p99']*1000:.2f} ms") if ttft["p99"] < 10 else print(f"  TTFT P99:     {ttft['p99']:.2f} ms")

    itl = metrics.get("itl", {})
    if itl.get("mean") is not None:
        print(f"  ITL Mean:     {itl['mean']*1000:.2f} ms") if itl["mean"] < 10 else print(f"  ITL Mean:     {itl['mean']:.2f} ms")
        if itl.get("p99"):
            print(f"  ITL P99:      {itl['p99']*1000:.2f} ms") if itl["p99"] < 10 else print(f"  ITL P99:      {itl['p99']:.2f} ms")

    tp = metrics.get("throughput", {})
    if tp.get("tokens_per_second"):
        print(f"\n  Throughput:   {tp['tokens_per_second']:.1f} tok/s")
    if tp.get("requests_per_second"):
        print(f"  RPS:          {tp['requests_per_second']:.2f} req/s")

    print(f"{'='*60}\n")


def run_multimodal_benchmark(
    target: str,
    model: str,
    image_dir: Optional[str] = None,
    max_requests: int = 10,
    processor: Optional[str] = None,
    output_dir: str = "./results/guidellm_multimodal",
    verbose: bool = False,
) -> Dict:
    """
    Run multimodal benchmark using GuideLLM with image+text prompts.

    For multimodal support, we generate a dataset of image+text prompts
    and feed them to GuideLLM via a local JSONL file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create multimodal dataset
    multimodal_prompts = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        for prompt in [
            "Describe a complex architectural blueprint in detail, including dimensions and materials.",
            "Analyze a scientific chart showing the relationship between GPU memory bandwidth and LLM inference throughput.",
            "Describe an MPI communication trace visualization showing point-to-point messages between 64 ranks.",
            "Explain what a flame graph of a CUDA application shows, identifying the hottest kernel.",
            "Analyze a timeline view from Nsight Systems showing CPU-GPU overlap in an inference pipeline.",
            "Describe the memory access pattern shown in a heatmap of cache line usage during matrix multiplication.",
            "Interpret a network topology diagram of a 4-node GPU cluster with NVLink and InfiniBand connections.",
            "Analyze a bar chart comparing TTFT, TPOT, and ITL metrics across three inference frameworks.",
            "Describe the performance bottleneck visible in a Gantt chart of pipeline-parallel LLM inference.",
            "Interpret a scatter plot of token latency vs sequence position for a 2048-token generation.",
        ]
    ]

    dataset_path = os.path.join(output_dir, "multimodal_prompts.jsonl")
    with open(dataset_path, "w") as f:
        for prompt in multimodal_prompts[:max_requests]:
            f.write(json.dumps(prompt) + "\n")

    # Run GuideLLM with the multimodal dataset
    return run_guidellm_benchmark(
        target=target,
        model=model,
        profile="synchronous",
        max_requests=max_requests,
        data=dataset_path,
        request_type="chat_completions",
        processor=processor,
        output_dir=output_dir,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="GuideLLM Benchmark Wrapper for LLM Inference Analysis"
    )
    parser.add_argument(
        "--target", type=str, default="http://localhost:11434/v1",
        help="OpenAI-compatible API endpoint (default: Ollama)"
    )
    parser.add_argument(
        "--model", type=str, default="llama2",
        help="Model name"
    )
    parser.add_argument(
        "--processor", type=str, default=None,
        help="HuggingFace processor/tokenizer ID (optional)"
    )
    parser.add_argument(
        "--profile", type=str, default="sweep",
        choices=["synchronous", "concurrent", "throughput", "constant", "poisson", "sweep"],
        help="Load profile"
    )
    parser.add_argument(
        "--rate", type=int, default=None,
        help="Rate value (meaning depends on profile)"
    )
    parser.add_argument(
        "--max-seconds", type=int, default=30,
        help="Maximum duration per benchmark"
    )
    parser.add_argument(
        "--max-requests", type=int, default=None,
        help="Maximum requests per benchmark"
    )
    parser.add_argument(
        "--data", type=str, default="prompt_tokens=256,output_tokens=128",
        help="Data specification"
    )
    parser.add_argument(
        "--request-type", type=str, default="chat_completions",
        choices=["chat_completions", "completions", "audio_transcription", "audio_translation"],
        help="API request type"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/guidellm",
        help="Output directory"
    )
    parser.add_argument(
        "--multimodal", action="store_true",
        help="Run multimodal benchmark instead of text-only"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Check / install GuideLLM
    if not check_guidellm_installed():
        print("GuideLLM not found. Attempting to install...")
        install_guidellm()
        if not check_guidellm_installed():
            print("ERROR: Could not install GuideLLM. Install manually:")
            print("  pip install guidellm[recommended] --break-system-packages")
            sys.exit(1)

    if args.multimodal:
        results = run_multimodal_benchmark(
            target=args.target,
            model=args.model,
            max_requests=args.max_requests or 10,
            processor=args.processor,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    else:
        results = run_guidellm_benchmark(
            target=args.target,
            model=args.model,
            profile=args.profile,
            rate=args.rate,
            max_seconds=args.max_seconds,
            max_requests=args.max_requests,
            data=args.data,
            request_type=args.request_type,
            processor=args.processor,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

    if "error" not in results:
        print_summary(results)


if __name__ == '__main__':
    main()
