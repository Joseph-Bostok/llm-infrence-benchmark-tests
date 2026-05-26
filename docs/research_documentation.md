# PInsight: Profiling and Benchmarking LLM Inference Pipelines
## Pre-Quantization Research Documentation

---

## 1. Research Problem

Large Language Model (LLM) inference is increasingly deployed in production systems, yet the performance characteristics of the inference pipeline remain poorly understood at the stage level. Practitioners face three key challenges:

1. **Memory Bandwidth Bottleneck** — Autoregressive decoding is fundamentally memory-bandwidth bound. Each token generation requires reading the entire model's weights from GPU DRAM, making decode throughput a function of memory bandwidth rather than compute capacity. On an NVIDIA A100, single-token decode operates at an arithmetic intensity of just **1 FLOP/byte** — 208× below the GPU's compute ridge point — meaning tensor cores sit nearly idle while the GPU waits on memory.

2. **Profiling Tool Accuracy** — Existing GPU profilers introduce significant overhead that distorts the very metrics they measure. Our experiments show that NVIDIA Nsight Systems adds **26.8% throughput overhead** while PyTorch Kineto adds **38.9%**. Without quantifying and accounting for these perturbations, performance analysis based on profiled runs can be misleading.

3. **Framework Abstraction Gap** — Production frameworks like vLLM optimize inference through techniques like PagedAttention and continuous batching, but abstract away the pipeline stages. This makes it impossible to attribute performance costs to specific inference phases (prefill vs. decode, attention vs. MLP) using framework-level APIs alone.

### 1.1 Research Objective

PInsight addresses these challenges by providing a **multi-layer profiling methodology** that:
- Decomposes inference into measurable pipeline stages (tokenization → prefill → decode → detokenization)
- Quantifies profiler overhead to establish trustworthy baselines
- Correlates application-level metrics with GPU kernel behavior
- Establishes pre-quantization baselines to validate the memory bandwidth hypothesis when weight precision is reduced

---

## 2. Tool Landscape & Market Position

### 2.1 Existing Tools on the Market

The LLM inference benchmarking space has several existing tools, each with distinct strengths and limitations:

| Tool | Focus | Strengths | Limitations |
|------|-------|-----------|-------------|
| **vLLM** | Production serving | PagedAttention, continuous batching, high throughput | Abstracts away pipeline stages; opaque to profiling |
| **NVIDIA Nsight Systems** | System-level profiling | Low overhead (26.8%), raw CUDA kernel traces | No semantic mapping to inference stages |
| **PyTorch Kineto** | Framework-level profiling | Maps Python operators → CUDA kernels, built-in | High overhead (38.9%), distorts production metrics |
| **GuideLLM** | Load testing & evaluation | Multi-profile load testing (sweep, poisson), fluidity index | No GPU-level profining, black-box metrics |
| **Etalon** | Holistic SLO evaluation | Fluidity index, SLO deadline tracking (TTFT/TBT) | Limited to application-level metrics |
| **LLMPerf / GenAI-Perf** | Throughput benchmarking | Quick throughput measurement, endpoint-agnostic | No stage-level decomposition |
| **LTTng / PInsight Tracing** | System-level tracing | CPU scheduling, page faults, context switches | Requires CTF trace infrastructure |

### 2.2 The Gap PInsight Fills

No single existing tool provides **stage-level pipeline decomposition with kernel attribution**. This is the gap PInsight fills:

- **Nsys/Kineto** see GPU kernels but can't tell you whether a kernel fired during prefill or decode
- **vLLM/GuideLLM/Etalon** see application metrics (TTFT, throughput) but can't attribute them to specific GPU operations
- **PInsight Pipeline Profiler** bridges these layers by using CUDA event timers + NVTX markers on bare HuggingFace models, enabling per-stage timing with optional kernel-level correlation through `--with-nsys`

### 2.3 Why Multiple Frameworks

We deliberately use different frameworks for different purposes:

| Framework | Role in PInsight | Rationale |
|-----------|-----------------|-----------|
| **vLLM** | Profiler overhead measurement | Represents real production workloads; PagedAttention and continuous batching give realistic throughput |
| **HuggingFace Transformers** | Pipeline stage decomposition | Bare model with no optimization — exposes individual stages (tokenization, prefill, decode, detokenization) for manual instrumentation |
| **GuideLLM** | Load profiling & SLO testing | Evaluates throughput under varying load patterns; provides fluidity index for streaming quality |
| **Etalon** | SLO-based evaluation | TTFT/TBT deadline compliance, complementary to raw metrics |
| **Ollama** | Local inference baseline | Accessible local inference for rapid development and comparison |

**Design rationale:** We use vLLM for overhead comparison because it represents real production workloads. We use bare HuggingFace for pipeline profiling because it exposes the individual inference stages that vLLM abstracts away. GuideLLM and Etalon provide complementary application-level metrics that contextualize the low-level findings.

---

## 3. Experimental Setup

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA A100-PCIE-40GB |
| **GPU Count** | 4 (single GPU used per experiment) |
| **Models** | Qwen/Qwen2.5-7B-Instruct (7.62B params), Qwen/Qwen2.5-14B-Instruct (14.77B params) |
| **Precision** | BFloat16 |
| **Framework** | vLLM (for profiler comparison), HuggingFace Transformers (for pipeline profiling) |
| **Profilers** | NVIDIA Nsight Systems (nsys), PyTorch Kineto, Custom Pipeline Profiler |
| **Benchmarking Tools** | GuideLLM, Etalon, Custom Ollama benchmark |
| **CUDA** | CUDA 12.x |

---

## 4. Profiling Tool Comparison

### 4.1 Tools Under Evaluation

We evaluated three profiling approaches, each operating at a different level of the inference stack:

| Tool | Level | Mechanism | What It Measures |
|------|-------|-----------|-----------------|
| **Nsight Systems (nsys)** | System | External process wrapping; intercepts CUDA driver calls via CUPTI | Raw CUDA kernels, memcpy, cuBLAS calls |
| **PyTorch Kineto** | Framework | `torch.profiler` hooks into PyTorch dispatcher | Python operators → CUDA kernel mapping |
| **Pipeline Profiler** | Application | Manual `torch.cuda.Event` timers + NVTX markers | Stage-level timing (tokenization, prefill, decode) |

### 4.2 Profiler Overhead Results

Both Nsight Systems and Kineto were benchmarked against an unprofiled baseline using identical workloads (Qwen2.5-7B, 10 requests, 128 max tokens):

| Metric | Baseline | Kineto (Profiled) | Kineto Overhead | Nsys (Profiled) | Nsys Overhead |
|--------|:--------:|:-----------------:|:---------------:|:---------------:|:-------------:|
| Throughput (tok/s) | 71.1 | 43.4 | **+38.9%** | 51.7 | **+26.8%** |
| Latency (s/req) | 1.80 | 2.95 | **+63.7%** | 2.46 | **+36.5%** |

**Key Finding:** Nsight Systems introduces **31% less throughput overhead** and **43% less latency overhead** than Kineto. This is because:
- **Kineto** hooks into the PyTorch dispatcher, adding instrumentation at every operator boundary (thousands of hooks per forward pass)
- **Nsys** intercepts at the CUDA driver level (CUPTI), which has fewer but coarser interception points
- **Nsys** runs as an **external process**, avoiding Python's GIL contention

> **Implication:** When measuring production inference performance, Nsys provides more accurate numbers. Kineto's higher overhead makes it better suited for debugging individual operator performance rather than measuring overall system throughput.

### 4.3 GPU Kernel Analysis

Both profilers identified the same kernel breakdown for Qwen2.5-7B inference:

| Category | Time (ms) | Calls | Kernels | % GPU Time |
|----------|:---------:|:-----:|:-------:|:----------:|
| GEMM / MatMul | 14,572.8 | 144,866 | 12 | **90.4%** |
| Elementwise | 379.1 | 47,747 | 8 | 2.4% |
| Attention | 335.6 | 38,976 | 2 | 2.1% |
| Normalization (RMSNorm) | 297.0 | 73,074 | 2 | 1.8% |
| Embedding (RoPE) | 174.4 | 35,896 | 1 | 1.1% |
| KV Cache | 123.1 | 35,868 | 1 | 0.8% |
| Softmax | 81.3 | 1,284 | 1 | 0.5% |

**Top CUDA Kernels:**

| # | Kernel | Calls | Total (ms) | % |
|---|--------|:-----:|:----------:|:-:|
| 1 | `ampere_bf16_s16816gemm_bf16_128x64` | 35,560 | 7,235.9 | 44.9% |
| 2 | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2` | 71,120 | 4,684.5 | 29.0% |
| 3 | `flash_fwd_splitkv_kernel` | 35,840 | 317.0 | 2.0% |
| 4 | `fused_add_rms_norm_kernel` | 71,792 | 292.8 | 1.8% |
| 5 | `rotary_embedding_kernel` | 35,896 | 174.4 | 1.1% |
| 6 | `reshape_and_cache_flash_kernel` | 35,868 | 123.1 | 0.8% |

**Key Finding:** GEMM operations consume **90.4% of GPU time**. This confirms that LLM inference is dominated by matrix multiplications — the weights-times-activations computation in the QKV projections and MLP layers. This is the primary motivation for quantization: reducing weight precision reduces the memory bandwidth required for these GEMMs.

### 4.4 Tool Capability Comparison

| Capability | Kineto | Nsys | Pipeline Profiler |
|-----------|:------:|:----:|:-----------------:|
| Profiler overhead measurement | ✅ 38.9% | ✅ 26.8% | — |
| TTFT (Time To First Token) | ❌ | ❌ | ✅ |
| Prefill vs Decode split | ❌ | ❌ | ✅ |
| Per-token decode latency | ❌ | ❌ | ✅ |
| GPU kernel names & timing | ✅ | ✅ | ❌ (unless `--with-nsys`) |
| Kernel-to-stage mapping | ❌ | ❌ | ✅ (with `--with-nsys`) |
| Python op → kernel trace | ✅ | ❌ | ❌ |
| Memory copy tracking | ❌ | ✅ | ❌ |
| Perfetto visualization | ✅ | ✅ | ✅ (with `--with-nsys`) |

---

## 5. Pipeline Stage Analysis

### 5.1 Inference Pipeline Stages

Using the custom Pipeline Profiler with bare HuggingFace Transformers (not vLLM), we decomposed inference into five measurable stages:

```
[CPU]  1. Tokenization     — BPE encoding of input text
[GPU]  2. H2D Transfer     — Host-to-Device tensor copy
[GPU]  3. Prefill           — Full attention over prompt (compute-bound)
[GPU]  4. Decode Loop       — Per-token autoregressive generation (memory-bound)
         ├─ Attention + KV Cache read
         ├─ MLP (GEMM)
         ├─ Logits projection
         └─ Sampling (argmax/top-k)
[CPU]  5. Detokenization    — Token IDs → output text
```

### 5.2 Qwen2.5-7B Results (7.62B Parameters)

| Stage | Time (ms) | % of Total | Detail |
|-------|:---------:|:----------:|--------|
| Tokenization | 0.6 | 0.0% | 13 tokens, CPU |
| H2D Transfer | 0.2 | 0.0% | 104 bytes |
| **Prefill** | **26.5** | **1.7%** | 2.03 ms/tok, TTFT = 27.3 ms |
| **Decode Loop** | **1,515.6** | **98.2%** | 64 tok, 42.2 tok/s |
| ├─ Forward (attn+mlp) | 1,512.5 | — | 23.6 ms/tok (P50) |
| └─ Sampling | 3.1 | — | 0.05 ms/tok |
| Detokenization | 0.2 | 0.0% | 330 chars, CPU |
| **Total** | **1,543.0** | **100%** | — |

**Memory:** 14,578 MB model + ~4 MB KV cache

### 5.3 Qwen2.5-14B Results (14.77B Parameters)

| Stage | Time (ms) | % of Total | Detail |
|-------|:---------:|:----------:|--------|
| Tokenization | 0.6 | 0.0% | 13 tokens, CPU |
| H2D Transfer | 0.2 | 0.0% | 104 bytes |
| **Prefill** | **43.7** | **0.8%** | 3.37 ms/tok, TTFT = 44.5 ms |
| **Decode Loop** | **5,265.7** | **99.2%** | 128 tok, 24.3 tok/s |
| ├─ Forward (attn+mlp) | 5,259.3 | — | 40.8 ms/tok (P50) |
| └─ Sampling | 6.4 | — | 0.05 ms/tok |
| Detokenization | 0.3 | 0.0% | 643 chars, CPU |
| **Total** | **5,310.5** | **100%** | — |

**Memory:** 28,327 MB model + ~27 MB KV cache

### 5.4 Model Size Scaling Analysis

| Metric | Qwen2.5-7B | Qwen2.5-14B | Scaling Factor |
|--------|:----------:|:-----------:|:--------------:|
| Parameters | 7.62B | 14.77B | 1.94× |
| VRAM | 14.6 GB | 28.3 GB | **1.94×** (linear) |
| TTFT | 27.3 ms | 44.5 ms | **1.63×** |
| Decode tok/s | 42.2 | 24.3 | **0.58×** (1.74× slower) |
| Per-token P50 | 23.6 ms | 40.8 ms | **1.73×** |
| Per-token P99 | 24.2 ms | 48.4 ms | **2.0×** |
| KV Cache | ~4 MB | ~27 MB | **6.75×** |

**Key Findings:**

1. **Decode scales worse than linearly** — 1.94× more params → 1.74× slower decode. This is because decode is memory-bandwidth bound, and the larger weight matrices exceed L2 cache, causing more DRAM reads per token.

2. **Prefill scales better** — 1.94× more params → only 1.63× slower prefill. Prefill is compute-bound (parallel over all prompt tokens), so it better utilizes the A100's tensor cores.

3. **P99 latency doubles for 14B** — indicating occasional memory pressure or cache thrashing that the 7B model doesn't experience.

4. **KV cache grows 6.75×** — disproportionate because the 14B model has both more layers and wider hidden dimensions.

---

## 6. The Memory Bandwidth Problem

### 6.1 Why Decode Is Memory-Bound

During autoregressive decoding, each token requires a full forward pass through the model, but only for a **single token** (batch size = 1). This means:

```
Arithmetic Intensity = FLOPs / Bytes Read

For Qwen2.5-7B decode (single token):
  FLOPs per token  ≈ 2 × 7.62B = 15.24 GFLOP
  Bytes read       ≈ 7.62B × 2 (BF16) = 15.24 GB
  Arithmetic Intensity = 15.24 GFLOP / 15.24 GB = 1 FLOP/byte

A100 Roofline:
  Peak compute     = 312 TFLOPS (BF16 tensor cores)
  Peak bandwidth   = 1.5 TB/s (HBM2e)
  Ridge point      = 312 / 1.5 = 208 FLOP/byte
```

At an arithmetic intensity of **1 FLOP/byte**, decode is **208× below the ridge point** — it's firmly in the memory-bandwidth-limited regime. The GPU's tensor cores are almost entirely idle during single-token decode.

### 6.2 Theoretical vs Observed Decode Throughput

```
Theoretical maximum decode rate (bandwidth-limited):
  Time per token = model_size_bytes / bandwidth
                 = 15.24 GB / 1.5 TB/s
                 = 10.2 ms/token → 98.4 tok/s

Observed (Qwen2.5-7B):  23.6 ms/token → 42.2 tok/s
Efficiency:              10.2 / 23.6 = 43.2% of theoretical bandwidth
```

The 43% bandwidth utilization is typical — overhead comes from KV cache reads, activation computation, kernel launch latency, and Python framework overhead.

### 6.3 Motivation for Quantization

Since decode is bandwidth-limited, **reducing model weight size directly increases decode throughput**:

| Precision | Weight Size | Theoretical tok/s | Expected Speedup |
|-----------|:----------:|:-----------------:|:----------------:|
| BF16 | 15.24 GB | 98.4 | 1.0× |
| INT8 | 7.62 GB | 196.8 | 2.0× |
| INT4 | 3.81 GB | 393.7 | 4.0× |

In practice, speedups are lower due to dequantization overhead and compute becoming the bottleneck at very low precisions. Expected real-world speedups: **INT8 ~1.3–1.5×**, **INT4 ~1.7–2.0×**.

---

## 7. Tool Ecosystem & Architecture

### 7.1 How the Tools Work Together

```
┌──────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  ┌────────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │ Pipeline        │  │ Kineto        │  │ Nsys           │  │
│  │ Profiler        │  │ Profiler      │  │ Profiler       │  │
│  │ (stage timing)  │  │ (op tracing)  │  │ (kernel trace) │  │
│  └───────┬────────┘  └──────┬────────┘  └───────┬────────┘  │
├──────────┼──────────────────┼───────────────────┼────────────┤
│          │   FRAMEWORK LAYER                    │            │
│  ┌───────▼────────┐  ┌──────▼────────┐          │            │
│  │ HuggingFace    │  │ vLLM          │          │            │
│  │ Transformers   │  │ (PagedAttn,   │          │            │
│  │ (bare model)   │  │  continuous   │          │            │
│  │                │  │  batching)    │          │            │
│  └───────┬────────┘  └──────┬────────┘          │            │
├──────────┼──────────────────┼───────────────────┼────────────┤
│          │   BENCHMARKING LAYER                  │            │
│  ┌───────▼────┐  ┌──────▼────┐  ┌──────────┐   │            │
│  │ Ollama     │  │ GuideLLM  │  │ Etalon   │   │            │
│  │ Benchmark  │  │ (load     │  │ (SLO     │   │            │
│  │ (local)    │  │  testing) │  │  metrics) │   │            │
│  └────────────┘  └───────────┘  └──────────┘   │            │
├──────────┼──────────────────────────────────────┼────────────┤
│          │   RUNTIME LAYER                      │            │
│  ┌───────▼──────────────────────────────────────▼────────┐   │
│  │              PyTorch / CUDA Runtime                    │   │
│  │  torch.cuda.Event ─── torch.profiler ─── CUPTI hooks  │   │
│  └───────────────────────────┬───────────────────────────┘   │
├──────────────────────────────┼───────────────────────────────┤
│          GPU HARDWARE        │                               │
│  ┌───────────────────────────▼───────────────────────────┐   │
│  │  A100: Tensor Cores + HBM2e (1.5 TB/s) + L2 Cache    │   │
│  │  GEMM kernels → cuBLAS (ampere_bf16_s16816gemm)       │   │
│  │  Attention    → Flash Attention (flash_fwd_splitkv)    │   │
│  │  Norm         → Custom CUDA (fused_add_rms_norm)       │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Cross-Layer Correlation (PInsight Correlator)

The PInsight Trace Correlator bridges all layers by:
1. Loading application-level token events (from benchmark results)
2. Loading runtime-level CUDA kernel traces (from Nsight Systems)
3. Loading system-level events (from LTTng traces — CPU scheduling, page faults)
4. Correlating events within a configurable time window to identify **which system-level events explain latency spikes**

This enables insights like: "Token 47 had a 3× latency spike caused by 12 page faults coinciding with KV cache expansion."

---

## 8. Next Steps: Quantization Study

The pre-quantization results establish clear baselines for:
- Stage-level timing (prefill vs decode split)
- Memory footprint (model weights + KV cache)
- Per-token decode latency distribution
- GPU kernel attribution

The next phase will:
1. **Apply quantization** (AWQ INT4, GPTQ INT4, bitsandbytes INT8/INT4)
2. **Re-run the pipeline profiler** with quantized models
3. **Compare stage-by-stage** how quantization affects each pipeline phase
4. **Measure accuracy impact** using perplexity and task-specific benchmarks
5. **Validate the bandwidth hypothesis** — does INT4 achieve the expected 1.7–2× decode speedup?

---

*Updated: March 30, 2026 | Hardware: NVIDIA A100-PCIE-40GB | Models: Qwen2.5-7B/14B-Instruct*
