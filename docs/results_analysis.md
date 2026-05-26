# PInsight Benchmark Results — Full Analysis
**Hardware:** NVIDIA A100-PCIE-40GB × 4 | **Date:** 2026-05-26  
**Models tested:** 15 across 4 size tiers | **Framework:** vLLM + Ollama

---

## 1. Cross-Tier Throughput Summary

All experiments used Ollama unless noted. vLLM numbers are from the baseline sweep (Section 4).

| Tier | Model | Params | Architecture | TTFT (ms) | TPOT (ms) | TPS | ITL P99 (ms) |
|---|---|---|---|---:|---:|---:|---:|
| **Tier 1 (7–9B)** | Mistral-7B | 7B | Dense + SWA | **66.5** | 6.74 | **148.4** | 8.89 |
| | Llama3-8B | 8B | Dense | 260.8 | 6.84 | 146.6 | 8.76 |
| | DeepSeek-R1-7B | 7B | Dense + CoT | 228.2 | 7.42 | 134.8 | 8.31 |
| | Qwen2.5-7B | 7B | Dense | 234.9 | 7.52 | 133.2 | 9.66 |
| | Llama3.1-8B | 8B | Dense | 254.0 | 7.60 | 131.8 | 11.39 |
| | Gemma2-9B | 9B | Dense | 251.7 | 10.39 | 96.4 | 13.14 |
| **Tier 2 (13–14B)** | Phi4-14B | 14B | Dense | 181.5 | 11.83 | 84.6 | 13.50 |
| | Qwen2.5-14B | 14B | Dense | 241.6 | 12.84 | 77.9 | 14.67 |
| | DeepSeek-R1-14B | 14B | Dense + CoT | 13,792 | 12.86 | 77.8 | 14.27 |
| **Tier 3 (27–32B)** | Gemma2-27B | 27B | Dense | 279.6 | 19.27 | 51.9 | 21.76 |
| | Qwen2.5-32B | 32B | Dense | 279.8 | 25.67 | 39.0 | 28.52 |
| | DeepSeek-R1-32B | 32B | Dense + CoT | 26,380 | 37.34 | 26.8 | 38.68 |
| **Tier 4 (70–72B)** | Qwen2.5-72B | 72B | Dense | 309.7 | 47.01 | **21.4** | — |
| | Llama3.3-70B | 70B | Dense | 400.5 | 66.86 | 15.0 | — |
| | DeepSeek-R1-70B | 70B | Dense + CoT | 44,463 | 67.83 | 14.8 | — |

> TTFT measured first-request cold. Subsequent warm requests are ~10× lower.  
> CoT models show extreme TTFT because they emit thousands of thinking tokens before the visible answer begins.

---

## 2. Throughput Scaling vs. Model Size

```
TPS (tokens/sec)
  150 │ ████  Mistral-7B (148)
      │ ████  Llama3-8B (147)
      │ ████  DeepSeek-R1-7B (135)
      │ ████  Qwen2.5-7B (133)
  100 │ ████  Llama3.1-8B (132)
      │ ████  Gemma2-9B (96)
      │
   80 │ ████  Phi4-14B (85)
      │ ████  Qwen2.5-14B (78)
      │ ████  DeepSeek-R1-14B (78)
      │
   50 │ ████  Gemma2-27B (52)
      │
   30 │ ████  Qwen2.5-32B (39)
      │ ████  DeepSeek-R1-32B (27)
      │
   20 │ ████  Qwen2.5-72B (21)
      │ ████  Llama3.3-70B (15)
   0  └──────────────────────────
         7B    14B    32B    72B
```

**Observed scaling:** TPS roughly halves every time parameters double, consistent with the memory-bandwidth-bound decode hypothesis. The 7→14B step is ~1.7× slower; 14→32B is ~2× slower; 32→72B is ~1.8× slower.

**Exception — Mistral-7B TTFT:** 66.5 ms vs. 228–261 ms for other 7B models. Mistral uses **Sliding Window Attention (SWA)** with a 4096-token window, which reduces prefill FLOP count and KV cache reads linearly with sequence length truncation.

**Exception — DeepSeek-R1 TTFT:** The CoT models have normal decode TPS but catastrophically high TTFT (13.8 s at 14B, 26.4 s at 32B, 44.5 s at 70B). This is not hardware-bound — it is the model generating tens of thousands of hidden reasoning tokens before the first visible response token. At 32B with ~37 ms/token TPOT, a 1000-token reasoning chain adds 37 seconds of apparent TTFT.

---

## 3. GPU Kernel Profiling — Qwen2.5-7B on A100

### 3.1 GPU Time by Operation Category (Nsight Systems)

| Category | GPU Time (ms) | Calls | % of Total |
|---|---:|---:|---:|
| **GEMM / MatMul** | **1,862.6** | 13,599 | **89.5%** |
| Activation | 48.5 | 3,369 | 2.3% |
| Attention (FlashAttn) | 44.1 | 6,656 | 2.1% |
| Normalization (RMSNorm) | 38.1 | 6,860 | 1.8% |
| Elementwise | 23.9 | 1,391 | 1.2% |
| Positional Encoding (RoPE) | 22.2 | 3,370 | 1.1% |
| Reduction | 16.0 | 3,493 | 0.8% |
| Softmax | 10.0 | 122 | 0.5% |
| Other | 15.8 | — | 0.8% |
| **Total** | **2,081.1** | **39,002** | 100% |

### 3.2 Top CUDA Kernels

| Rank | Kernel | Calls | Total (ms) | % |
|---|---|---:|---:|---:|
| 1 | `ampere_bf16_s16816gemm_bf16_128x64` | 3,258 | 665.0 | 32.0% |
| 2 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2` | 6,515 | 434.5 | 20.9% |
| 3 | `ampere_bf16_s16816gemm_bf16_128x256` | 56 | 337.6 | 16.2% |
| 4 | `ampere_bf16_s16816gemm_bf16_256x128` | 86 | 172.7 | 8.3% |
| 5 | `act_and_mul_kernel` (SwiGLU) | 3,258 | ~48.5 | 2.3% |
| 6 | `flash_fwd_splitkv_kernel` | ~6,656 | 44.1 | 2.1% |

### 3.3 Prefill vs. Decode Time Split

| Phase | Time (ms) | % of Inference |
|---|---:|---:|
| Prefill (prompt processing) | 549.6 | 27% |
| **Decode (token generation)** | **1,475.8** | **71%** |
| Other (scheduling, overhead) | 55.7 | 3% |

**Key finding:** Decode dominates at 71% of total inference time despite generating far fewer tokens per unit time than prefill processes prompt tokens. This confirms the memory-bandwidth-bound decode hypothesis — each decode step reads the entire 15 GB model from HBM for a single token.

### 3.4 Roofline Analysis

```
Arithmetic Intensity (FLOP/byte) for Qwen2.5-7B single-token decode:
  FLOPs per token  = 2 × 7.62B params = 15.24 GFLOP
  Bytes read       = 7.62B × 2 (BF16) = 15.24 GB
  Intensity        = 1.0 FLOP/byte

A100 ridge point   = 312 TFLOPS / 1.5 TB/s = 208 FLOP/byte

Decode is 208× below the compute ridge point — tensor cores sit idle.
Tensor core utilization during decode: ~0.5%
```

**Bandwidth efficiency:** Theoretical peak at 1.5 TB/s → 98.4 tok/s. Observed: ~42 tok/s (HuggingFace bare model). Efficiency = **43%**. The remaining 57% is overhead: KV cache reads, kernel launch latency, sampling ops, Python/CUDA runtime.

---

## 4. vLLM Baseline Parameter Sweep

**Model:** Qwen/Qwen2.5-7B-Instruct | **Hardware:** A100-PCIE-40GB (single GPU)

### 4.1 Default Configuration (Baseline)

| Workload | TPS | TTFT (ms) | P99 TTFT (ms) |
|---|---:|---:|---:|
| Dialogue | 72.6 | 261 | 731 |
| RAG | 78.1 | 80 | 183 |
| Code | 79.0 | 70 | 155 |
| Reasoning | 77.1 | 109 | 159 |

> Default: `gpu_memory_utilization=0.90, block_size=16, max_model_len=8192`

### 4.2 Live Sweep Results — Config 1 (Preliminary)

*Config 1 of 13 complete. Full sweep in progress (~156 min estimated).*

| Workload | TPS | TTFT (ms) | Notes |
|---|---:|---:|---|
| Dialogue | 72.5 | 267 | Cold first token 746ms, warm ~28ms |
| RAG | 78.4 | 82 | |
| Code | 78.7 | 78 | |
| Reasoning | *running* | — | |

**Observation:** First-request TTFT is ~20–25× higher than subsequent requests (746ms vs. 27–29ms). This is KV cache cold-start — the first request must fill the KV cache from scratch. Subsequent requests reuse memory-resident activations.

### 4.3 Parameter Space Under Sweep

The quick sweep varies one parameter at a time from the default, covering:

| Parameter | Values Under Test | Hypothesis |
|---|---|---|
| `gpu_memory_utilization` | 0.80, 0.85, **0.90**, 0.95 | Higher = more KV cache space = fewer evictions |
| `block_size` | 8, **16**, 32 | Smaller blocks = less fragmentation waste |
| `max_model_len` | 4096, **8192**, 16384, 32768 | Larger = more memory pressure, higher TTFT |
| `enable_prefix_caching` | False, **True** | Hit rate depends on shared prefix fraction |
| `max_num_batched_tokens` | 2048, 4096, **8192**, 16384 | Larger = better GPU utilization during prefill |

*Full results will be added when sweep completes.*

---

## 5. Key Findings & Implications

### Finding 1: Memory bandwidth is the hard wall
Decode throughput is determined entirely by A100 HBM bandwidth (1.5 TB/s). GEMM kernels consume 89.5% of GPU time but tensor cores are <1% utilized — all time is spent waiting for weights to arrive from HBM. This means:
- Adding more CUDA cores would not improve decode speed
- Quantization (INT8, INT4) directly reduces model bytes → directly increases TPS

### Finding 2: CoT models have a TTFT problem, not a throughput problem
DeepSeek-R1 models have similar TPOT to same-size dense models (7.42 ms/tok vs. 6.84–7.60 ms/tok for other 7B models). The extreme TTFT is the reasoning trace itself, not hardware. Solutions: output streaming (show reasoning tokens to user), speculative decoding, or distilled dense models.

### Finding 3: Architecture choices dominate at the same parameter count
At 7B parameters, Mistral's SWA cuts TTFT to 66.5 ms (vs 228–261 ms for dense models) — a 3.4–3.9× improvement with no hardware change. GQA in Qwen2.5 (4 KV heads vs 32 query heads) keeps KV cache at ~4 MB for 64-token sequences — 8× smaller than standard multi-head attention.

### Finding 4: The TPS–TTFT tradeoff across tiers

| Target Use Case | Best Model | Reason |
|---|---|---|
| Lowest latency (chatbot) | Mistral-7B | SWA = 66ms TTFT |
| Highest raw throughput | Llama3-8B | 146.6 TPS |
| Best 14B throughput | Phi4-14B | 84.6 TPS (vs 77.9 for Qwen2.5-14B) |
| Best 70B throughput | Qwen2.5-72B | 21.4 TPS (vs 15.0 for Llama3.3-70B) |
| Reasoning quality | DeepSeek-R1 (any) | CoT traces, accept TTFT cost |

### Finding 5: KV cache is not yet the bottleneck at short contexts
At 64-token sequences, KV cache for Qwen2.5-7B = ~4 MB vs. model weights = 15.2 GB (0.026%). KV cache only becomes a significant bandwidth competitor at >8K token contexts (see `paper_notes.md` Section 3.2 for the bandwidth split analysis). This validates focusing the next research phase on KV optimization for long-context workloads.

---

## 6. Next Steps

| Priority | Action | Expected Impact |
|---|---|---|
| **1** | Complete vLLM baseline sweep (13 configs × 4 workloads) | Identifies optimal native vLLM config per workload |
| **2** | Run Qwen2.5-7B with AWQ INT4 quantization | Expected ~1.7–2.0× decode speedup (bandwidth theory) |
| **3** | Pipeline profiler at multiple sequence lengths (64→32768 tokens) | Find the KV cache "latency cliff" |
| **4** | Implement H2O eviction on HuggingFace bare model | Validate 50% cache budget with <5% quality loss |
| **5** | KIVI 2-bit KV quantization on Qwen2.5-7B | Validate 2.6× KV memory reduction claim |

---

*Hardware: NVIDIA A100-PCIE-40GB × 4 | Models: vLLM 0.21.0, PyTorch 2.11.0+cu130 | Analysis: PInsight*
