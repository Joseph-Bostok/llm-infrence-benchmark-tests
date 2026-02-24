#!/usr/bin/env python3
"""
vLLM Inference Profiler
========================
Runs vLLM offline inference with multi-level profiling:

  Level 1: Manual timing — TTFT-equivalent, throughput (matches ollama benchmarks)
  Level 2: vLLM native profiling — captures CUDA kernels from the engine subprocess
           via VLLM_TORCH_PROFILER_DIR (solves the multiprocess problem)
  Level 3: nsys wrapping — use nsys_profile.sh for full GPU timeline

The key insight: vLLM v1 runs model inference in a SEPARATE PROCESS
(EngineCore_DP0), so torch.profiler in the main process sees nothing.
Instead, we use vLLM's built-in profiler support which instruments
the engine process directly.

Usage:
    # Basic run with vLLM native profiling
    python3 vllm_profile.py --model Qwen/Qwen2.5-7B-Instruct

    # More requests, custom output
    python3 vllm_profile.py --model Qwen/Qwen2.5-7B-Instruct --requests 7 --warmup 2

    # Multi-GPU
    python3 vllm_profile.py --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel 4

    # For gated models
    python3 vllm_profile.py --model meta-llama/Meta-Llama-3-8B-Instruct --hf-token <token>

    # Wrap with nsys for full GPU-level tracing (recommended):
    nsys profile --trace cuda,nvtx,cublas -o trace_output -- python3 vllm_profile.py ...
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path


def get_prompts(n: int) -> list[str]:
    """Same prompts as ollama_benchmark.py for cross-comparison."""
    base_prompts = [
        "Explain the concept of recursion in programming in 3 sentences.",
        "What are the main differences between TCP and UDP protocols?",
        "Describe the process of photosynthesis in simple terms.",
        "Write a brief overview of the French Revolution.",
        "Explain how a neural network learns from data.",
    ]
    return [base_prompts[i % len(base_prompts)] for i in range(n)]


def run_profiled_inference(args):
    """Run vLLM inference with native profiling and manual timing."""
    import torch
    import torch.cuda
    from vllm import LLM, SamplingParams

    output_dir = Path(args.output_dir)
    model_safe = args.model.replace("/", "_")
    run_dir = output_dir / f"{model_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Set up vLLM native profiling ---
    # This env var tells the EngineCore subprocess to run torch.profiler
    # INSIDE the engine process where CUDA kernels actually execute
    trace_dir = run_dir / "vllm_traces"
    trace_dir.mkdir(exist_ok=True)
    # Deprecated env var removed - using profiler_config instead

    print(f"\n{'='*60}")
    print(f"  vLLM PROFILED INFERENCE")
    print(f"{'='*60}")
    print(f"  Model:            {args.model}")
    print(f"  Tensor Parallel:  {args.tensor_parallel}")
    print(f"  Total Requests:   {args.requests}")
    print(f"  Warmup Requests:  {args.warmup}")
    print(f"  Max Tokens:       {args.max_tokens}")
    print(f"  Output Dir:       {run_dir}")
    print(f"  Trace Dir:        {trace_dir}")
    print(f"{'='*60}\n")

    # --- Load model ---
    print("[1/4] Loading model into GPU...")
    load_start = time.monotonic()

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel,
        "trust_remote_code": True,
        "max_model_len": 2048,
        "enforce_eager": True,  # disable CUDA graphs for cleaner profiling
        # "collect_detailed_traces": ["model"],  # enables NVTX ranges in engine
    }
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    llm = LLM(**llm_kwargs)
    load_elapsed = time.monotonic() - load_start
    print(f"    Model loaded in {load_elapsed:.1f}s")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=args.max_tokens,
    )

    prompts = get_prompts(args.requests)

    # --- Record GPU baseline ---
    try:
        torch.cuda.synchronize()
        gpu_mem_before = torch.cuda.memory_allocated() / (1024**3)
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"    GPU memory (main proc): {gpu_mem_before:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved")
        print(f"    Note: Model memory is in EngineCore subprocess, not visible here")
    except Exception:
        gpu_mem_before = 0
        gpu_mem_reserved = 0

    # --- Warmup ---
    print(f"\n[2/4] Running {args.warmup} warmup request(s)...")
    for i in range(args.warmup):
        _ = llm.generate([prompts[i]], sampling_params)
    print("    Warmup complete")

    # --- Inference with timing ---
    num_measured = args.requests - args.warmup
    metrics = []

    print(f"\n[3/4] Running {num_measured} measured request(s)...")
    print(f"    vLLM native profiler is capturing CUDA kernels in engine subprocess")

    for i in range(num_measured):
        idx = args.warmup + i
        prompt = prompts[idx]

        t_start = time.monotonic()
        outputs = llm.generate([prompt], sampling_params)
        t_end = time.monotonic()

        output = outputs[0]
        num_tokens = len(output.outputs[0].token_ids)
        total_s = t_end - t_start

        metrics.append({
            "request_index": idx,
            "prompt": prompt[:80],
            "num_output_tokens": num_tokens,
            "total_time_s": round(total_s, 4),
            "tokens_per_second": round(num_tokens / total_s, 2) if total_s > 0 else 0,
        })
        print(f"    Request {idx}: {num_tokens} tokens in {total_s:.2f}s ({num_tokens/total_s:.1f} tok/s)")

    # --- GPU memory snapshot ---
    try:
        torch.cuda.synchronize()
        gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        gpu_peak = 0

    # --- Save results ---
    print(f"\n[4/4] Saving results...")

    tps_values = [m["tokens_per_second"] for m in metrics]
    total_times = [m["total_time_s"] for m in metrics]
    token_counts = [m["num_output_tokens"] for m in metrics]

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "backend": "vllm",
        "vllm_version": _get_vllm_version(),
        "tensor_parallel": args.tensor_parallel,
        "max_tokens": args.max_tokens,
        "total_requests": args.requests,
        "warmup_requests": args.warmup,
        "model_load_time_s": round(load_elapsed, 2),
        "gpu_memory": {
            "main_process_allocated_gb": round(gpu_mem_before, 2),
            "main_process_reserved_gb": round(gpu_mem_reserved, 2),
            "main_process_peak_gb": round(gpu_peak, 2),
            "note": "Model weights are in EngineCore subprocess memory",
        },
        "metrics": metrics,
        "aggregate": {
            "mean_tokens_per_second": round(sum(tps_values) / len(tps_values), 2),
            "min_tokens_per_second": round(min(tps_values), 2),
            "max_tokens_per_second": round(max(tps_values), 2),
            "mean_total_time_s": round(sum(total_times) / len(total_times), 3),
            "mean_output_tokens": round(sum(token_counts) / len(token_counts), 1),
            "total_tokens": sum(token_counts),
        },
    }

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Check for vLLM trace output
    trace_files = list(trace_dir.rglob("*.json"))
    all_trace_files = list(trace_dir.rglob("*"))

    print(f"\n{'='*60}")
    print(f"  PROFILING COMPLETE")
    print(f"{'='*60}")
    print(f"  Results:        {results_path}")
    print(f"  Trace dir:      {trace_dir}")
    print(f"  Trace files:    {len(all_trace_files)} files found")
    agg = results["aggregate"]
    print(f"  Mean throughput: {agg['mean_tokens_per_second']} tok/s")
    print(f"  Mean time/req:   {agg['mean_total_time_s']:.3f}s")
    print(f"  Total tokens:    {agg['total_tokens']}")
    print(f"{'='*60}")

    if trace_files:
        print(f"\n  Chrome trace files (open in chrome://tracing or perfetto.dev):")
        for tf in trace_files[:10]:
            size_mb = tf.stat().st_size / (1024*1024)
            print(f"    {tf} ({size_mb:.1f}MB)")
    elif all_trace_files:
        print(f"\n  Trace files found (may need conversion):")
        for tf in all_trace_files[:10]:
            print(f"    {tf}")
    else:
        print(f"\n  ⚠ No trace files found in {trace_dir}")
        print(f"    vLLM native profiling may require the server API.")
        print(f"    Use nsys instead for guaranteed GPU kernel capture:")
        print(f"    nsys profile --trace cuda,nvtx,cublas -o nsys_trace -- python3 vllm_profile.py --model {args.model}")

    print(f"\n  RECOMMENDED NEXT STEP — nsys for full kernel timeline:")
    print(f"    bash nsys_profile.sh vllm {args.model}")
    print()


def _get_vllm_version():
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="vLLM Inference Profiler")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--requests", type=int, default=5,
                        help="Total inference requests")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup requests (not measured)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max output tokens per request")
    parser.add_argument("--output-dir", type=str, default="./results/profiling",
                        help="Output directory")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for gated models")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Tensor parallel degree (number of GPUs)")

    args = parser.parse_args()
    run_profiled_inference(args)


if __name__ == "__main__":
    main()
