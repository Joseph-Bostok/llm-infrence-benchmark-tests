#!/usr/bin/env python3
"""
Etalon LLM Inference Benchmark Wrapper
Provides a simplified interface to run Etalon benchmarks against Ollama or other OpenAI-compatible servers.

Etalon metrics:
- TTFT: Time to First Token
- TBT: Time Between Tokens
- TPOT: Time Per Output Token
- Fluidity Index: Novel metric for measuring streaming smoothness
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add etalon to path
ETALON_PATH = Path(__file__).parent.parent / "etalon"
sys.path.insert(0, str(ETALON_PATH))


# Mapping from Ollama model names to HuggingFace model identifiers
OLLAMA_TO_HF_MODEL = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama2": "NousResearch/Llama-2-7b-chat-hf",
    "llama2:7b": "NousResearch/Llama-2-7b-chat-hf",
    "llama2:13b": "NousResearch/Llama-2-13b-chat-hf",
    "llama3": "NousResearch/Meta-Llama-3-8B-Instruct",
    "llama3:8b": "NousResearch/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "codellama": "codellama/CodeLlama-7b-Instruct-hf",
    "phi": "microsoft/phi-2",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "gemma": "google/gemma-7b-it",
    "gemma:2b": "google/gemma-2b-it",
    "qwen": "Qwen/Qwen-7B-Chat",
    "qwen2": "Qwen/Qwen2-7B-Instruct",
}


def get_hf_model_name(ollama_model: str) -> str:
    """Convert Ollama model name to HuggingFace model identifier."""
    # Check exact match
    if ollama_model in OLLAMA_TO_HF_MODEL:
        return OLLAMA_TO_HF_MODEL[ollama_model]

    # Check prefix match (e.g., "tinyllama:latest" -> "tinyllama")
    base_name = ollama_model.split(":")[0].lower()
    if base_name in OLLAMA_TO_HF_MODEL:
        return OLLAMA_TO_HF_MODEL[base_name]

    # Return as-is if not found (might be a valid HF model)
    return ollama_model


def setup_environment(api_base: str, api_key: str = "ollama"):
    """Set up environment variables for Etalon."""
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    # Disable wandb by default
    os.environ["WANDB_MODE"] = "disabled"


def run_etalon_benchmark(
    model: str,
    api_base: str,
    output_dir: str,
    max_requests: int = 50,
    num_clients: int = 1,
    concurrent_requests: int = 1,
    ttft_deadline: float = 0.5,
    tbt_deadline: float = 0.05,
    qps: float = 0.5,
    timeout: int = 300,
    trace_file: str = None,
    verbose: bool = False
) -> dict:
    """
    Run Etalon benchmark with specified configuration.

    Args:
        model: Model name (e.g., 'llama2', 'mistral')
        api_base: API endpoint (e.g., 'http://localhost:11434/v1')
        output_dir: Directory to store results
        max_requests: Maximum number of requests to complete
        num_clients: Number of parallel clients
        concurrent_requests: Concurrent requests per client
        ttft_deadline: TTFT deadline in seconds
        tbt_deadline: TBT deadline in seconds
        qps: Queries per second (for Poisson distribution)
        timeout: Benchmark timeout in seconds
        trace_file: Optional trace file for request patterns
        verbose: Print verbose output

    Returns:
        Dictionary with benchmark results
    """
    setup_environment(api_base)

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get HuggingFace model name for tokenizer (separate from API model name)
    hf_tokenizer = get_hf_model_name(model)
    print(f"API model: {model}")
    print(f"Tokenizer: {hf_tokenizer}")

    # Build command - use Ollama model name for API, HF model for tokenizer
    cmd = [
        sys.executable, "-m", "etalon.run_benchmark",
        "--client_config_model", model,
        "--client_config_tokenizer", hf_tokenizer,
        "--max_completed_requests", str(max_requests),
        "--timeout", str(timeout),
        "--client_config_num_clients", str(num_clients),
        "--client_config_num_concurrent_requests_per_client", str(concurrent_requests),
        "--metrics_config_output_dir", output_dir,
        "--request_interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(qps),
        "--deadline_config_ttft_deadline", str(ttft_deadline),
        "--deadline_config_tbt_deadline", str(tbt_deadline),
    ]

    # Add trace file if provided, otherwise use synthetic
    if trace_file and Path(trace_file).exists():
        cmd.extend([
            "--request_length_generator_config_type", "trace",
            "--trace_request_length_generator_config_trace_file", trace_file,
        ])
    else:
        cmd.extend([
            "--request_length_generator_config_type", "uniform",
            "--uniform_request_length_generator_config_min_tokens", "50",
            "--uniform_request_length_generator_config_max_tokens", "200",
        ])

    print("="*60)
    print("ETALON BENCHMARK")
    print("="*60)
    print(f"Model: {model}")
    print(f"API Base: {api_base}")
    print(f"Max Requests: {max_requests}")
    print(f"Clients: {num_clients} x {concurrent_requests} concurrent")
    print(f"QPS: {qps}")
    print(f"Output: {output_dir}")
    print("-"*60)

    if verbose:
        print(f"Command: {' '.join(cmd)}")

    print("\nRunning benchmark...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=timeout + 60,
            cwd=str(ETALON_PATH)
        )

        if result.returncode != 0:
            print(f"Benchmark failed with return code {result.returncode}")
            if not verbose and result.stderr:
                print(f"Error: {result.stderr[:500]}")
            return {"error": result.stderr}

        if verbose and result.stdout:
            print(result.stdout)

    except subprocess.TimeoutExpired:
        print("Benchmark timed out")
        return {"error": "timeout"}
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return {"error": str(e)}

    # Load and return results
    results = load_etalon_results(output_dir)
    print_etalon_summary(results)

    return results


def load_etalon_results(output_dir: str) -> dict:
    """Load results from Etalon output directory."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": output_dir,
        "metrics": {}
    }

    output_path = Path(output_dir)

    # Look for results files
    for results_file in output_path.glob("*.json"):
        try:
            with open(results_file) as f:
                data = json.load(f)
                results["metrics"][results_file.stem] = data
        except Exception as e:
            print(f"Could not load {results_file}: {e}")

    # Look for CSV results
    for csv_file in output_path.glob("*.csv"):
        results["metrics"][csv_file.stem + "_csv"] = str(csv_file)

    return results


def print_etalon_summary(results: dict) -> None:
    """Print summary of Etalon results."""
    print("\n" + "="*60)
    print("ETALON RESULTS SUMMARY")
    print("="*60)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    metrics = results.get("metrics", {})

    if not metrics:
        print("No metrics collected")
        return

    for name, data in metrics.items():
        if isinstance(data, dict):
            print(f"\n{name}:")
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"{name}: {data}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Etalon LLM Benchmark Wrapper')
    parser.add_argument('--model', default='llama2',
                        help='Model name')
    parser.add_argument('--api-base', default='http://localhost:11434/v1',
                        help='OpenAI-compatible API base URL')
    parser.add_argument('--output-dir', '-o', default='./results/etalon',
                        help='Output directory for results')
    parser.add_argument('--max-requests', type=int, default=50,
                        help='Maximum requests to complete')
    parser.add_argument('--num-clients', type=int, default=1,
                        help='Number of parallel clients')
    parser.add_argument('--concurrent', type=int, default=1,
                        help='Concurrent requests per client')
    parser.add_argument('--qps', type=float, default=0.5,
                        help='Queries per second')
    parser.add_argument('--ttft-deadline', type=float, default=0.5,
                        help='TTFT deadline in seconds')
    parser.add_argument('--tbt-deadline', type=float, default=0.05,
                        help='TBT deadline in seconds')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Benchmark timeout in seconds')
    parser.add_argument('--trace-file', type=str,
                        help='Trace file for request patterns')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    run_etalon_benchmark(
        model=args.model,
        api_base=args.api_base,
        output_dir=args.output_dir,
        max_requests=args.max_requests,
        num_clients=args.num_clients,
        concurrent_requests=args.concurrent,
        qps=args.qps,
        ttft_deadline=args.ttft_deadline,
        tbt_deadline=args.tbt_deadline,
        timeout=args.timeout,
        trace_file=args.trace_file,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
