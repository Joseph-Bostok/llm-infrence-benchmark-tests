#!/usr/bin/env python3
"""
pipeline_profiler.py — Stage-by-Stage LLM Inference Profiler
=============================================================
Breaks inference into measurable pipeline stages and records
per-stage latency, GPU memory, and kernel attribution:

  [CPU]  1. Tokenization
  [GPU]  2. H2D Transfer
  [GPU]  3. Prefill (full prompt attention)
  [GPU]  4. Decode loop (per-token):
            4a. Attention + KV cache
            4b. MLP (GEMM)
            4c. Logits + Softmax
            4d. Sampling
  [CPU]  5. Detokenization

Uses bare HuggingFace transformers (not vLLM) for precise stage
control.  Wraps each stage in NVTX markers so nsys traces show
labeled regions in Perfetto.

Usage:
  source venv/bin/activate
  python3 pipeline_profiler.py --model Qwen/Qwen2.5-7B-Instruct
  python3 pipeline_profiler.py --model Qwen/Qwen2.5-7B-Instruct --with-nsys

Author: PInsight Benchmark Suite
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Colors ────────────────────────────────────────────────────
C = {
    "h": "\033[1;36m", "ok": "\033[0;32m", "w": "\033[1;33m",
    "e": "\033[0;31m", "d": "\033[0;90m", "v": "\033[0;37m",
    "b": "\033[1;34m", "r": "\033[0m",
}


def banner(text, char="═"):
    w = 70
    print(f"\n{C['h']}{char * w}\n  {text}\n{char * w}{C['r']}")


def section(text):
    print(f"\n{C['b']}── {text} {'─' * max(0, 60 - len(text))}{C['r']}")


def kv(key, val, unit="", indent=4):
    print(f"{' ' * indent}{C['d']}{key}:{C['r']} {C['v']}{val}{C['r']} {C['d']}{unit}{C['r']}")


# ── Prompts ───────────────────────────────────────────────────

PROMPTS = [
    "Explain the concept of attention mechanisms in transformers in detail.",
    "What are the main differences between GPT and BERT architectures?",
    "Describe how KV-cache optimization works in autoregressive language models.",
    "Compare tensor parallelism and pipeline parallelism for distributed inference.",
    "Explain the process of tokenization and its impact on model performance.",
    "How does flash attention reduce memory usage while maintaining accuracy?",
    "Describe the role of positional encoding in transformer architectures.",
    "What optimizations does vLLM use for efficient batch inference?",
]


def get_prompts(n):
    return [PROMPTS[i % len(PROMPTS)] for i in range(n)]


# ── GPU helpers ───────────────────────────────────────────────

def gpu_mem_mb():
    """Current GPU memory allocated in MB."""
    import torch
    return torch.cuda.memory_allocated() / (1024 ** 2)


def gpu_mem_reserved_mb():
    import torch
    return torch.cuda.memory_reserved() / (1024 ** 2)


def cuda_event_pair():
    """Create a pair of CUDA events for precise GPU timing."""
    import torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def event_elapsed_ms(start, end):
    """Get elapsed time between two CUDA events in ms."""
    import torch
    torch.cuda.synchronize()
    return start.elapsed_time(end)


# ── Stage profiler ────────────────────────────────────────────

def profile_single_request(model, tokenizer, prompt, max_new_tokens, device,
                           request_idx=0):
    """
    Profile a single inference request through all pipeline stages.
    Returns a dict with per-stage timing and memory data.
    """
    import torch
    import torch.cuda.nvtx as nvtx

    result = {
        "request_index": request_idx,
        "prompt": prompt[:80],
        "prompt_length_chars": len(prompt),
        "stages": {},
        "memory_checkpoints": {},
        "decode_per_token": [],
    }

    mem_before = gpu_mem_mb()
    result["memory_checkpoints"]["before_inference"] = round(mem_before, 1)

    # ─── Stage 1: Tokenization (CPU) ─────────────────────────
    nvtx.range_push("1_tokenization")
    t0_tok = time.perf_counter()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    t1_tok = time.perf_counter()
    nvtx.range_pop()

    prompt_tokens = input_ids.shape[1]
    result["prompt_tokens"] = prompt_tokens
    result["stages"]["tokenization"] = {
        "time_ms": round((t1_tok - t0_tok) * 1000, 3),
        "tokens": prompt_tokens,
        "device": "cpu",
    }

    # ─── Stage 2: H2D Transfer ───────────────────────────────
    nvtx.range_push("2_h2d_transfer")
    s_h2d, e_h2d = cuda_event_pair()
    s_h2d.record()
    input_ids_gpu = input_ids.to(device)
    e_h2d.record()
    nvtx.range_pop()

    h2d_ms = event_elapsed_ms(s_h2d, e_h2d)
    result["stages"]["h2d_transfer"] = {
        "time_ms": round(h2d_ms, 3),
        "tensor_shape": list(input_ids.shape),
        "bytes": input_ids.numel() * input_ids.element_size(),
        "device": "gpu",
    }
    result["memory_checkpoints"]["after_h2d"] = round(gpu_mem_mb(), 1)

    # ─── Stage 3: Prefill (first forward pass) ───────────────
    nvtx.range_push("3_prefill")
    s_pf, e_pf = cuda_event_pair()
    torch.cuda.synchronize()
    s_pf.record()
    with torch.no_grad():
        outputs = model(input_ids_gpu, use_cache=True)
    e_pf.record()
    nvtx.range_pop()

    prefill_ms = event_elapsed_ms(s_pf, e_pf)
    past_key_values = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]

    result["stages"]["prefill"] = {
        "time_ms": round(prefill_ms, 3),
        "prompt_tokens": prompt_tokens,
        "ms_per_token": round(prefill_ms / max(prompt_tokens, 1), 3),
        "device": "gpu",
    }
    result["memory_checkpoints"]["after_prefill"] = round(gpu_mem_mb(), 1)
    result["ttft_ms"] = round(
        result["stages"]["tokenization"]["time_ms"]
        + h2d_ms + prefill_ms, 3
    )

    # ─── Stage 4: Decode (autoregressive loop) ───────────────
    nvtx.range_push("4_decode_loop")
    generated_ids = []
    total_decode_ms = 0
    total_attn_ms = 0
    total_sample_ms = 0
    eos_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        token_metrics = {"step": step}

        # 4a. Sampling from logits of previous step
        nvtx.range_push(f"4d_sampling_step{step}")
        s_samp, e_samp = cuda_event_pair()
        s_samp.record()
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        e_samp.record()
        nvtx.range_pop()

        sample_ms = event_elapsed_ms(s_samp, e_samp)
        token_metrics["sampling_ms"] = round(sample_ms, 3)
        total_sample_ms += sample_ms

        token_id = next_token.item()
        generated_ids.append(token_id)

        if token_id == eos_id:
            token_metrics["eos"] = True
            result["decode_per_token"].append(token_metrics)
            break

        # 4b. Forward pass (attention + MLP + logits) for next token
        nvtx.range_push(f"4abc_forward_step{step}")
        s_fwd, e_fwd = cuda_event_pair()
        s_fwd.record()
        with torch.no_grad():
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
        e_fwd.record()
        nvtx.range_pop()

        fwd_ms = event_elapsed_ms(s_fwd, e_fwd)
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        token_metrics["forward_ms"] = round(fwd_ms, 3)
        token_metrics["total_ms"] = round(fwd_ms + sample_ms, 3)
        total_decode_ms += fwd_ms
        total_attn_ms += fwd_ms  # In transformer, forward = attn + mlp + logits

        result["decode_per_token"].append(token_metrics)

    nvtx.range_pop()  # 4_decode_loop

    n_generated = len(generated_ids)
    result["generated_tokens"] = n_generated

    # Decode stage summary
    result["stages"]["decode"] = {
        "time_ms": round(total_decode_ms + total_sample_ms, 3),
        "forward_ms": round(total_decode_ms, 3),
        "sampling_ms": round(total_sample_ms, 3),
        "tokens_generated": n_generated,
        "ms_per_token": round(
            (total_decode_ms + total_sample_ms) / max(n_generated, 1), 3
        ),
        "tokens_per_second": round(
            n_generated / max((total_decode_ms + total_sample_ms) / 1000, 0.001), 2
        ),
        "device": "gpu",
    }
    result["memory_checkpoints"]["after_decode"] = round(gpu_mem_mb(), 1)

    # KV cache memory estimate
    kv_mem = gpu_mem_mb() - mem_before
    result["memory_checkpoints"]["kv_cache_estimate_mb"] = round(max(kv_mem, 0), 1)

    # ─── Stage 5: Detokenization (CPU) ────────────────────────
    nvtx.range_push("5_detokenization")
    t0_det = time.perf_counter()
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    t1_det = time.perf_counter()
    nvtx.range_pop()

    result["stages"]["detokenization"] = {
        "time_ms": round((t1_det - t0_det) * 1000, 3),
        "output_chars": len(output_text),
        "device": "cpu",
    }

    # ─── Total & breakdown ────────────────────────────────────
    total_ms = sum(s["time_ms"] for s in result["stages"].values())
    result["total_inference_ms"] = round(total_ms, 3)

    result["stage_breakdown_pct"] = {}
    for name, stage in result["stages"].items():
        result["stage_breakdown_pct"][name] = round(
            stage["time_ms"] / max(total_ms, 0.001) * 100, 1
        )

    result["output_text_preview"] = output_text[:200]

    # Free KV cache
    del past_key_values, outputs, next_logits, next_token, input_ids_gpu
    torch.cuda.empty_cache()

    return result


# ── Pretty print ──────────────────────────────────────────────

def print_request_summary(req, idx):
    """Print a formatted summary of one profiled request."""
    section(f"Request {idx}: {req['prompt_tokens']} prompt → "
            f"{req['generated_tokens']} generated tokens")

    stages = req["stages"]
    total = req["total_inference_ms"]
    pcts = req["stage_breakdown_pct"]

    print(f"\n    {C['d']}{'Stage':<25} {'Time (ms)':>12} {'%':>8} {'Detail':>30}{C['r']}")
    print(f"    {C['d']}{'─' * 78}{C['r']}")

    # Tokenization
    s = stages["tokenization"]
    print(f"    {C['v']}{'1. Tokenization':<25}{C['r']}"
          f" {s['time_ms']:>12.3f} {pcts['tokenization']:>7.1f}%"
          f" {C['d']}{s['tokens']} tokens (CPU){C['r']}")

    # H2D
    s = stages["h2d_transfer"]
    print(f"    {C['v']}{'2. H2D Transfer':<25}{C['r']}"
          f" {s['time_ms']:>12.3f} {pcts['h2d_transfer']:>7.1f}%"
          f" {C['d']}{s['bytes']} bytes{C['r']}")

    # Prefill
    s = stages["prefill"]
    print(f"    {C['v']}{'3. Prefill':<25}{C['r']}"
          f" {s['time_ms']:>12.3f} {pcts['prefill']:>7.1f}%"
          f" {C['d']}{s['ms_per_token']:.2f} ms/tok, TTFT={req['ttft_ms']:.1f}ms{C['r']}")

    # Decode
    s = stages["decode"]
    print(f"    {C['v']}{'4. Decode Loop':<25}{C['r']}"
          f" {s['time_ms']:>12.3f} {pcts['decode']:>7.1f}%"
          f" {C['d']}{s['tokens_generated']} tok, {s['tokens_per_second']:.1f} tok/s{C['r']}")
    print(f"    {C['d']}{'   ├─ Forward (attn+mlp)':<25}{C['r']}"
          f" {s['forward_ms']:>12.3f}")
    print(f"    {C['d']}{'   └─ Sampling':<25}{C['r']}"
          f" {s['sampling_ms']:>12.3f}")

    # Detokenization
    s = stages["detokenization"]
    print(f"    {C['v']}{'5. Detokenization':<25}{C['r']}"
          f" {s['time_ms']:>12.3f} {pcts['detokenization']:>7.1f}%"
          f" {C['d']}{s['output_chars']} chars (CPU){C['r']}")

    print(f"    {C['d']}{'─' * 78}{C['r']}")
    print(f"    {C['v']}{'TOTAL':<25}{C['r']}"
          f" {total:>12.3f} {100.0:>7.1f}%")

    # Memory
    mem = req["memory_checkpoints"]
    print(f"\n    {C['d']}Memory: before={mem['before_inference']:.0f} MB → "
          f"after_prefill={mem['after_prefill']:.0f} MB → "
          f"after_decode={mem['after_decode']:.0f} MB | "
          f"KV cache ≈ {mem['kv_cache_estimate_mb']:.0f} MB{C['r']}")


def print_aggregate(all_results):
    """Print aggregate stats across all requests."""
    section("Aggregate Pipeline Breakdown")

    stage_names = ["tokenization", "h2d_transfer", "prefill", "decode", "detokenization"]
    stage_labels = {
        "tokenization": "1. Tokenization",
        "h2d_transfer": "2. H2D Transfer",
        "prefill": "3. Prefill",
        "decode": "4. Decode Loop",
        "detokenization": "5. Detokenization",
    }

    # Compute means
    means = {}
    for sn in stage_names:
        times = [r["stages"][sn]["time_ms"] for r in all_results]
        means[sn] = sum(times) / len(times) if times else 0

    total_mean = sum(means.values())

    print(f"\n    {C['d']}{'Stage':<25} {'Mean (ms)':>12} {'%':>8}{C['r']}")
    print(f"    {C['d']}{'─' * 48}{C['r']}")
    for sn in stage_names:
        pct = means[sn] / total_mean * 100 if total_mean > 0 else 0
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"    {C['v']}{stage_labels[sn]:<25}{C['r']}"
              f" {means[sn]:>12.1f} {pct:>7.1f}%"
              f" {C['d']}{bar}{C['r']}")
    print(f"    {C['d']}{'─' * 48}{C['r']}")
    print(f"    {C['v']}{'TOTAL':<25}{C['r']} {total_mean:>12.1f} {100.0:>7.1f}%")

    # TTFT and decode tok/s
    ttfts = [r["ttft_ms"] for r in all_results]
    decode_tps = [r["stages"]["decode"]["tokens_per_second"] for r in all_results]
    print(f"\n    {C['d']}Mean TTFT:          {sum(ttfts)/len(ttfts):.1f} ms{C['r']}")
    print(f"    {C['d']}Mean Decode tok/s:  {sum(decode_tps)/len(decode_tps):.1f}{C['r']}")

    # Prefill vs decode split
    pf_pct = means["prefill"] / total_mean * 100 if total_mean > 0 else 0
    dc_pct = means["decode"] / total_mean * 100 if total_mean > 0 else 0
    print(f"    {C['d']}Prefill vs Decode:  {pf_pct:.0f}% / {dc_pct:.0f}%{C['r']}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Stage Profiler for LLM Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pipeline_profiler.py --model Qwen/Qwen2.5-7B-Instruct --requests 3
  python3 pipeline_profiler.py --model Qwen/Qwen2.5-7B-Instruct --with-nsys
        """,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output-dir", default="./results/pipeline")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--with-nsys", action="store_true",
                        help="Re-execute under nsys profile for kernel traces")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    # Hidden inner flag for nsys re-exec
    parser.add_argument("--_inner", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # If --with-nsys and not already inner, re-exec under nsys
    if args.with_nsys and not args._inner:
        nsys_bin = shutil.which("nsys")
        if not nsys_bin:
            print(f"{C['e']}nsys not found in PATH{C['r']}")
            sys.exit(1)

        model_safe = args.model.replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (Path(args.output_dir) / f"{model_safe}_{ts}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "nsys_trace"

        inner_cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--_inner",
            "--model", args.model,
            "--requests", str(args.requests),
            "--warmup", str(args.warmup),
            "--max-tokens", str(args.max_tokens),
            "--output-dir", str(out_dir),
            "--dtype", args.dtype,
        ]
        if args.hf_token:
            inner_cmd.extend(["--hf-token", args.hf_token])

        nsys_cmd = [
            nsys_bin, "profile",
            "--trace", "cuda,nvtx,cublas",
            "--cuda-memory-usage", "true",
            "--output", str(report_path),
            "--force-overwrite", "true",
            "--kill", "sigterm",
            "--",
        ] + inner_cmd

        print(f"{C['h']}Launching under nsys...{C['r']}")
        print(f"{C['d']}Report: {report_path}.nsys-rep{C['r']}")
        subprocess.run(nsys_cmd, cwd=str(out_dir))
        print(f"\n{C['ok']}nsys trace: {report_path}.nsys-rep{C['r']}")
        print(f"{C['d']}View: nsys-ui {report_path}.nsys-rep{C['r']}")
        return

    # ── Direct execution (or inner mode) ──

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Output dir
    if args._inner:
        output_dir = Path(args.output_dir).resolve()
    else:
        model_safe = args.model.replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (Path(args.output_dir) / f"{model_safe}_{ts}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    banner("PIPELINE STAGE PROFILER", "▓")
    kv("Model", args.model)
    kv("Requests", f"{args.requests} (warmup: {args.warmup})")
    kv("Max Tokens", str(args.max_tokens))
    kv("Dtype", args.dtype)
    kv("Device", str(device))
    kv("Output", str(output_dir))
    if torch.cuda.is_available():
        kv("GPU", torch.cuda.get_device_name(0))
        kv("GPU count", str(torch.cuda.device_count()))
        free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        total_gb = torch.cuda.mem_get_info()[1] / (1024**3)
        kv("GPU Memory", f"{free_gb:.1f} / {total_gb:.1f} GB free")

    # ── Load model ────────────────────────────────────────────
    section("Loading Model")
    t0_load = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    t1_load = time.perf_counter()
    kv("Load time", f"{t1_load - t0_load:.1f}", "s")
    kv("Parameters", f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}", "B")
    kv("GPU after load", f"{gpu_mem_mb():.0f}", "MB")

    # ── Warmup ────────────────────────────────────────────────
    if args.warmup > 0:
        section(f"Warmup ({args.warmup} requests)")
        prompts = get_prompts(args.warmup)
        for i, p in enumerate(prompts):
            profile_single_request(model, tokenizer, p, min(args.max_tokens, 16),
                                   device, i)
        print(f"    {C['ok']}Warmup complete{C['r']}")
        torch.cuda.synchronize()

    # ── Profile measured requests ─────────────────────────────
    banner("PROFILING INFERENCE PIPELINE")
    measured_prompts = get_prompts(args.requests)[args.warmup:]
    if not measured_prompts:
        measured_prompts = get_prompts(max(args.requests - args.warmup, 1))

    all_results = []
    for i, prompt in enumerate(measured_prompts):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        req_result = profile_single_request(
            model, tokenizer, prompt, args.max_tokens, device, i
        )
        all_results.append(req_result)
        print_request_summary(req_result, i)

    # ── Aggregate ─────────────────────────────────────────────
    print_aggregate(all_results)

    # ── Decode per-token latency stats ────────────────────────
    section("Decode Per-Token Latency Distribution")
    all_fwd_ms = []
    for r in all_results:
        for t in r["decode_per_token"]:
            if "forward_ms" in t:
                all_fwd_ms.append(t["forward_ms"])

    if all_fwd_ms:
        all_fwd_ms.sort()
        p50 = all_fwd_ms[len(all_fwd_ms) // 2]
        p99 = all_fwd_ms[int(len(all_fwd_ms) * 0.99)]
        kv("Min", f"{min(all_fwd_ms):.2f}", "ms")
        kv("P50", f"{p50:.2f}", "ms")
        kv("P99", f"{p99:.2f}", "ms")
        kv("Max", f"{max(all_fwd_ms):.2f}", "ms")
        kv("Mean", f"{sum(all_fwd_ms)/len(all_fwd_ms):.2f}", "ms")

    # ── Save results ──────────────────────────────────────────
    section("Saving Results")

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "dtype": args.dtype,
        "max_tokens": args.max_tokens,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "requests": all_results,
        "aggregate": {
            "mean_total_ms": round(
                sum(r["total_inference_ms"] for r in all_results)
                / len(all_results), 1
            ),
            "mean_ttft_ms": round(
                sum(r["ttft_ms"] for r in all_results)
                / len(all_results), 1
            ),
            "mean_decode_tok_s": round(
                sum(r["stages"]["decode"]["tokens_per_second"] for r in all_results)
                / len(all_results), 1
            ),
            "stage_means_ms": {
                sn: round(
                    sum(r["stages"][sn]["time_ms"] for r in all_results)
                    / len(all_results), 1
                )
                for sn in ["tokenization", "h2d_transfer", "prefill",
                           "decode", "detokenization"]
            },
        },
        "decode_latency_stats": {
            "min_ms": round(min(all_fwd_ms), 2) if all_fwd_ms else 0,
            "p50_ms": round(p50, 2) if all_fwd_ms else 0,
            "p99_ms": round(p99, 2) if all_fwd_ms else 0,
            "max_ms": round(max(all_fwd_ms), 2) if all_fwd_ms else 0,
            "mean_ms": round(sum(all_fwd_ms)/len(all_fwd_ms), 2) if all_fwd_ms else 0,
        },
    }

    results_path = output_dir / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    kv("Results", str(results_path))

    # ── Summary ───────────────────────────────────────────────
    banner("PIPELINE PROFILING COMPLETE", "▓")
    agg = results_data["aggregate"]
    kv("Results", str(results_path))
    kv("Mean Total", f"{agg['mean_total_ms']:.1f}", "ms")
    kv("Mean TTFT", f"{agg['mean_ttft_ms']:.1f}", "ms")
    kv("Mean Decode", f"{agg['mean_decode_tok_s']:.1f}", "tok/s")
    print()

    # Cleanup
    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
