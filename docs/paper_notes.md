# KV Cache Research — Paper Notes

---

## Papers

| # | Paper | arXiv | Venue | Category |
|---|-------|-------|-------|----------|
| 1 | A Survey on Large Language Model Acceleration based on KV Cache Management | 2412.19442 | 2024 | Survey |
| 2 | KV Cache Optimization Strategies for Scalable and Efficient LLM Inference | 2603.20397 | 2026 | Survey |
| 3 | Efficient Memory Management for LLM Serving with PagedAttention (vLLM) | 2309.06180 | SOSP 2023 | System |
| 4 | FlashAttention: Fast and Memory-Efficient Exact Attention | 2205.14135 | NeurIPS 2022 | Attention |
| 5 | FlashAttention-2: Better Parallelism and Work Partitioning | 2307.08691 | ICLR 2024 | Attention |
| 6 | H2O: Heavy-Hitter Oracle for Efficient Generative Inference | 2306.14048 | NeurIPS 2023 | Eviction |
| 7 | SnapKV: LLM Knows What You Are Looking for Before Generation | 2404.14469 | 2024 | Eviction |
| 8 | KIVI: Tuning-Free Asymmetric 2bit Quantization for KV Cache | 2402.02750 | ICML 2024 | Quantization |
| 9 | MiniKV: 2-Bit Layer-Discriminative KV Cache | 2411.18077 | 2024 | Hybrid |
| 10 | ShadowKV: KV Cache in Shadows for Long-Context Inference | 2410.21465 | 2024 | Offloading |
| 11 | LMCache: Efficient KV Cache Layer for Enterprise LLM Inference | 2025 | 2025 | System |
| 12 | Mooncake: KV Cache-Centric Disaggregated Architecture | 2407.00079 | 2024 | System |
| 13 | GQA: Training Generalized Multi-Query Transformer Models | 2305.13245 | 2023 | Architecture |
| 14 | Splitwise: Efficient Generative LLM Inference Using Phase Splitting | 2311.18677 | 2023 | System |
| 15 | Dissecting Runtime Performance of LLM Training, Fine-tuning, and Inference | 2311.03687 | 2023 | Profiling |

---

## The Core Problem

LLMs are slow and memory-intensive at inference time. This comes down to three things:

1. **Attention is O(n²) computation** — every token attends to every previous token
2. **KV cache grows O(n) with sequence length** — each new token adds a key+value vector to the cache, and in practice this gets huge
3. **Longer context blows up both memory and latency** — more tokens = more cache = more bandwidth consumed per decode step

### Why KV Cache Is The Bottleneck

During generation, every token produces a **key vector** and a **value vector**. These get stored in the KV cache. Future tokens attend to all previous KV pairs — instead of recomputing attention from scratch, we reuse the cached keys and values to avoid redundant computation.

This is faster than recomputing, but it means:
- The cache grows with every token generated
- Every decode step reads the entire cache (all previous KV pairs)
- Memory bandwidth becomes the true bottleneck — the GPU spends more time reading cache than computing

**This is what makes long-context inference expensive.** Not the compute — the memory reads.

---

## Three Layers of KV Cache Optimization

The survey papers [1, 2] organize the entire optimization space into three layers:

### Layer 1: Token-Level Optimizations

**Goal:** Use less memory and compute per token.
**Tradeoff:** Quality can drop if you're too aggressive.

| Technique | What It Does | Key Papers |
|-----------|-------------|------------|
| **KV Pruning** | Drop the least important tokens from the cache entirely | H2O [6] |
| **KV Selection** | Only keep the most useful tokens per attention head | SnapKV [7] |
| **KV Merging** | Combine similar tokens into one compressed representation | — |
| **Quantization** | Reduce precision of cached keys/values (FP16 → INT4 → INT2) | KIVI [8], MiniKV [9] |
| **Low-Rank Compression** | Factorize the KV matrices into smaller representations | ShadowKV [10] (keys) |

**How pruning works (H2O [6]):**
- Only ~5% of tokens are "heavy hitters" that get most of the attention
- Keep those heavy hitters + recent tokens
- Evict everything else
- Result: 29× throughput gain, minimal quality loss

**How selection works (SnapKV [7]):**
- Look at the last few tokens of the prompt ("observation window")
- Each attention head consistently focuses on the same tokens
- Keep only those important tokens per head
- Result: 3.6× speed, 8.2× memory efficiency at 16K context

**How quantization works (KIVI [8]):**
- Keys and values have different distributions — they need different quantization strategies
- Keys → per-channel quantization (fixed outlier channels)
- Values → per-token quantization (no consistent outlier pattern)
- Result: 2-bit precision, 2.6× memory reduction, near-baseline quality

**Hybrid approach (MiniKV [9]):**
- Combines 2-bit quantization + token eviction
- Uses a layer-discriminative policy — different layers get different budgets
- Key insight: **more tokens at lower precision beats fewer tokens at higher precision**
- Result: 86% KV cache compression, 98.5% accuracy recovery

---

### Layer 2: Model-Level Optimizations

**Goal:** Reduce how much KV cache the model needs to look at.
**Approach:** Change how the model itself uses the cache — these are architectural changes.

| Technique | What It Does | Key Papers |
|-----------|-------------|------------|
| **Sparse Attention** | Only attend to a subset of positions instead of all previous tokens | — |
| **Sliding Window Attention** | Only attend to the most recent N tokens | Mistral |
| **Chunking** | Break long sequences into chunks, process attention within chunks | ChunkKV |
| **Attention Redesign (GQA)** | Share KV heads across multiple query heads → fewer KV pairs to store | GQA [13] |
| **Attention Redesign (MQA)** | Single KV head shared by all query heads → minimal KV cache | — |
| **Multi-Head Latent Attention (MLA)** | Compress KV into a latent space before caching | DeepSeek-V3 |

**How GQA works [13]:**
- Standard multi-head attention: 32 query heads, 32 KV heads → full KV cache
- GQA: 32 query heads, 4 KV heads → 8× smaller KV cache
- Qwen2.5-7B uses GQA with 4 KV heads — this is why our KV cache was only ~4 MB for 64 tokens

**Why this matters for our profiling:**
- GQA is already baked into the models we test
- Our profiler measures the *result* of these architectural choices
- The KV cache size we observe (4 MB at 64 tokens for 7B) is already 8× smaller than it would be with standard multi-head attention

---

### Layer 3: System-Level Optimizations

**Goal:** Make KV cache more manageable at scale. Don't change the model — change the infrastructure.

| Technique | What It Does | Key Papers |
|-----------|-------------|------------|
| **Memory Paging** | Break KV cache into fixed-size blocks, allocate on demand (like OS virtual memory) | vLLM / PagedAttention [3] |
| **Memory Offloading** | Move KV cache between GPU → CPU → disk depending on usage | ShadowKV [10], LMCache [11] |
| **Scheduling / Batching** | Group requests to maximize cache reuse and GPU utilization | vLLM continuous batching |
| **Multi-GPU Cache Sharing** | Share KV caches across GPUs or across separate prefill/decode machines | Mooncake [12], LMCache [11] |
| **Prefix Caching** | Reuse KV cache for common prompt prefixes across different requests | SGLang (RadixAttention) |
| **Prefill-Decode Disaggregation** | Run prefill on compute-optimized GPUs, decode on bandwidth-optimized GPUs | Splitwise [14], Mooncake [12] |

**How PagedAttention works (vLLM [3]):**
- Traditional: pre-allocate contiguous memory for max sequence length → huge waste
- PagedAttention: break KV cache into 16-token blocks, allocate blocks on demand
- Blocks don't need to be contiguous in GPU memory
- Result: near-zero memory waste, 2–4× throughput improvement
- This is the baseline that everyone builds on

**How offloading works (ShadowKV [10]):**
- Keep compressed keys on GPU (small footprint)
- Offload full values to CPU memory (large but cheaper)
- During decode, reconstruct only the sparse KV pairs actually needed
- Overlap key reconstruction with value fetching to hide latency
- Result: 6× larger batches, 3× throughput on A100

**How disaggregation works (Splitwise [14], Mooncake [12]):**
- Prefill = compute-bound (process all prompt tokens in parallel)
- Decode = memory-bound (read entire model per token)
- These have different hardware requirements → run them on different GPU pools
- Transfer KV cache from prefill cluster to decode cluster over network
- Result: better resource utilization, lower cost per token

---

## How FlashAttention Fits In [4, 5]

FlashAttention is not a KV cache optimization — it's an attention computation optimization. But it's critical background:

- Standard attention materializes the full N×N attention matrix in GPU HBM → O(n²) memory
- FlashAttention uses **tiling**: computes attention in small blocks that fit in GPU SRAM (on-chip)
- Never writes the full attention matrix to HBM — recomputes as needed
- Result: O(n) memory, 2–4× faster, exact same output

**Why this matters for our profiling:**
- `flash_fwd_splitkv_kernel` in our nsys traces = FlashAttention doing the KV cache reads
- It's only 2% of GPU time because it's extremely well-optimized
- The 90.4% spent in GEMM is the weight reads — the actual bottleneck

---

## Where We Fit In

All of these papers either:
- **Propose an optimization** (eviction, quantization, paging, offloading)
- **Monitor aggregate metrics** (cache usage %, preemption count)

**Nobody provides per-token, per-layer KV cache profiling.** That's our gap:
- Track KV cache growth per token
- Measure bandwidth split: weight reads vs KV cache reads
- Identify when KV cache pressure causes latency spikes
- Score token importance to guide eviction decisions
- Show practitioners which optimization to apply for their specific workload

---

## Deep Dive: Cache Compression Techniques

### The Fundamental Tradeoff

Every compression technique makes the same trade: **use less memory → risk losing information**. The question is *how* you lose information and *how much* it matters for the downstream task.

There are four ways to compress a KV cache:

```
1. Drop tokens entirely          (Eviction)    → lose some tokens, keep full precision
2. Reduce numeric precision      (Quantization) → keep all tokens, lose some precision
3. Combine similar tokens        (Merging)      → fewer tokens, each is an average
4. Factor the matrices           (Low-Rank)     → keep all tokens, approximate the math
```

The key insight from MiniKV [9]: **these are not alternatives — they're complementary.** The best results come from combining them. Specifically, keeping more tokens at lower precision (quant + selection) beats keeping fewer tokens at higher precision (aggressive eviction alone).

### How Each Compression Method Works Under the Hood

#### Eviction: Which Tokens Matter?

The core question is: given a KV cache with N tokens, which K tokens should we keep? Every eviction method needs an **importance score** for each token.

**H2O's scoring [6]:**
```
importance(token_i) = cumulative_attention_score(token_i)
                    = Σ over all layers, all heads: attention_weight[head][query → token_i]
```
Keep the top-K by cumulative attention + always keep the most recent W tokens (sliding window).

Problem: attention scores change as generation progresses. A token that was unimportant at step 10 might become critical at step 100. H2O handles this by continuously updating scores, but it's still a greedy heuristic.

**SnapKV's scoring [7]:**
```
For each attention head h:
  1. Take the last P tokens of the prompt (the "observation window")
  2. Compute attention scores from those P tokens to all previous tokens
  3. The consistently high-scoring tokens are the "important" ones for this head
  4. Each head keeps its own selected set of tokens
```
Why this works: the authors found that attention patterns are **stable** — the tokens a head focuses on during the observation window are the same ones it will focus on during generation. So you can predict importance before generation starts.

Why it beats H2O: per-head selection catches head-specific patterns that global scoring misses.

**Attention Sinks (StreamingLLM):**
```
A critical discovery: the first few tokens in any sequence get disproportionate 
attention regardless of their content. This is a mathematical artifact of softmax:

  softmax forces attention weights to sum to 1
  → when no token is semantically relevant, the model "dumps" weight on token 0
  → token 0 becomes an "attention sink"

Therefore: ALWAYS keep the first 4 tokens + a sliding window of recent tokens.
Simple, but prevents the catastrophic failure that pure sliding-window eviction causes.
```

#### Quantization: How Low Can You Go?

**Standard approach (per-tensor):**
```python
# Quantize entire KV cache tensor to INT8
scale = max(abs(tensor)) / 127
quantized = round(tensor / scale).to(int8)
# Dequantize: tensor_approx = quantized * scale
```
Problem: outlier values dominate the scale, crushing the precision of normal values.

**KIVI's approach [8] — asymmetric per-channel/per-token:**
```python
# Keys have outlier CHANNELS (certain dimensions always have large values)
# → quantize per-channel (each channel gets its own scale)
for channel in range(head_dim):
    scale_k[channel] = max(abs(key_cache[:, channel])) / max_int
    key_quant[:, channel] = round(key_cache[:, channel] / scale_k[channel])

# Values have outlier TOKENS (certain tokens have large values across all dimensions)  
# → quantize per-token (each token gets its own scale)
for token in range(seq_len):
    scale_v[token] = max(abs(value_cache[token, :])) / max_int
    value_quant[token, :] = round(value_cache[token, :] / scale_v[token])
```
Why asymmetric: the authors found that K and V have fundamentally different distributions. Keys have a few channels with consistently large magnitudes across all tokens. Values have a few tokens with large magnitudes across all channels. Using the wrong axis for quantization destroys information.

**MiniKV's approach [9] — sub-channel + layer-discriminative:**
```
Layer budget allocation:
  - Not all layers are equally sensitive to compression
  - Early layers + late layers need more cache budget
  - Middle layers can be compressed more aggressively
  - Each layer gets a different (tokens_to_keep, bit_width) allocation

Sub-channel key quantization:
  - KIVI uses per-channel → good but wastes bits on non-outlier regions within a channel
  - MiniKV splits each channel into sub-groups of 8-16 values
  - Each sub-group gets its own scale → better precision where it matters
```

#### Merging: Combine Instead of Drop

Instead of evicting token 47 entirely, merge it with a similar token:
```
token_merged = weighted_average(token_47, token_48)
```
This preserves more information than eviction but adds compute overhead.

**KVCrush (Intel, 2025):**
- Converts token KV vectors into compact **binary representations**
- Clusters tokens by head behavior using lightweight clustering
- Picks the most representative token from each cluster
- 4× compression with <1% accuracy drop on LongBench
- <0.5% inference latency overhead
- Compatible with vLLM, FlashAttention, and other eviction methods

#### Low-Rank: Factor the Matrices

**ShadowKV [10]:**
```
Key observation: pre-RoPE keys are low-rank across the sequence dimension.

Instead of storing full keys: K ∈ R^(seq_len × head_dim)
Factor into: K ≈ U × S × V^T where U ∈ R^(seq_len × r), r << head_dim

Store the low-rank factors on GPU (small), offload values to CPU (large).
During decode, reconstruct only the sparse KV pairs that matter.
```

---

## Deep Dive: Newer Methods (2024–2025)

### ChunkKV — Semantic-Preserving Compression

**Problem with existing eviction:** dropping individual tokens can break semantic coherence. If you keep token 5 but drop tokens 3, 4, 6, 7 — you lose the context around token 5.

**Solution:** group tokens into contiguous **chunks** (e.g., 16-token segments). Keep or drop entire chunks.

```
Before (per-token eviction):
  tokens: [The][cat][sat][on][the][mat][and][then][the][dog][came][in][...]
  kept:   [The]     [sat]          [mat]     [then]     [dog]     [in]
  → fragmented, lost context around each kept token

After (chunk eviction):
  chunks: [The cat sat on] [the mat and then] [the dog came in] [...]
  kept:   [The cat sat on]                    [the dog came in]
  → coherent semantic units preserved
```

Additional trick: **layer-wise index reuse** — adjacent layers select nearly the same chunks, so you can reuse the selection indices instead of recomputing importance scores for each layer. This cuts the eviction compute overhead.

### CurDKV — Value-Guided Selection

**Problem with attention-score-based methods:** H2O and SnapKV rank tokens by attention scores (Q × K^T), but ignore the value vectors. The final output is `softmax(QK^T) × V` — a token can have high attention but a near-zero value vector, meaning it barely affects the output.

**Solution:** use **CUR decomposition** (a matrix factorization technique) to identify which tokens best preserve the full attention output:

```
Attention output = softmax(Q × K^T) × V
                 = A × V

CUR decomposition of A:
  - Select columns (tokens) that best reconstruct A × V
  - Use "leverage scores" as the importance metric
  - Leverage score considers both A (attention) and V (value contribution)

Result: tokens are ranked by their actual contribution to the output, 
not just by how much attention they receive.
```

Why it's better: at 70–90% compression, CurDKV maintains accuracy where attention-only methods start to degrade. The value-guidance catches tokens that matter for the output even if they don't dominate attention scores.

### R-KV — Redundancy-Aware Compression for Reasoning

**Problem for reasoning models:** chain-of-thought (CoT) traces are long and repetitive. The model says "Let me reconsider..." or "Going back to..." multiple times. Standard eviction keeps these redundant tokens because they have high attention scores (the model attends to its own reflection).

**Solution:** add a **redundancy penalty** to the importance score:

```
score(token_i) = importance(token_i) × (1 - redundancy(token_i))

redundancy(token_i) = max_similarity(token_i, all_other_kept_tokens)
```

If a token is semantically near-identical to another token already in the cache, its score gets penalized. This naturally deduplicates reasoning traces.

Why it matters: reasoning models like DeepSeek-R1 can generate 10K+ token CoT traces. R-KV compresses these to ~20% with minimal reasoning quality loss.

---

## How to Evaluate: Benchmarks

### Quality Benchmarks (Does compression hurt output quality?)

| Benchmark | What It Tests | Typical Usage |
|-----------|-------------|---------------|
| **LongBench** | 6 task categories across long-context: single/multi-doc QA, summarization, few-shot, code, synthetic | Standard for KV compression papers. Test accuracy at 4K–16K tokens. |
| **RULER** | Synthetic tasks: needle-in-haystack, multi-key retrieval, variable tracking | Stress-tests retrieval at extreme lengths (32K–128K). Catches failures that LongBench misses. |
| **Needle-in-a-Haystack** | Single fact retrieval buried in long context | Simple but effective. If compression drops the "needle" token, accuracy goes to zero. |
| **Perplexity (PPL)** | Language model quality on held-out text | Quick sanity check. Higher PPL = worse model. But PPL doesn't catch retrieval failures well. |
| **MMLU / HellaSwag / ARC** | Short-context accuracy | Baseline sanity check — compression shouldn't hurt short-context tasks. |

### Performance Benchmarks (Does compression actually speed things up?)

| Metric | What It Measures | How We Measure It |
|--------|-----------------|-------------------|
| **Decode throughput (tok/s)** | Tokens generated per second | Pipeline profiler: end-to-end decode time / tokens generated |
| **TTFT (Time to First Token)** | Latency until first generated token | Pipeline profiler: tokenization + H2D + prefill time |
| **Peak VRAM** | Maximum GPU memory used | `torch.cuda.max_memory_allocated()` |
| **KV cache size** | Bytes used by cached keys and values | `sum(k.numel() * k.element_size() for k, v in past_key_values)` |
| **Per-token P50/P99 latency** | Decode latency distribution | Pipeline profiler: per-step CUDA event timing |
| **Compression ratio** | Original KV size / compressed KV size | Direct measurement |

### Why Some Methods Beat Others

The evaluation tells a clear story:

```
At 16K context length, ~50% KV cache budget:

                Accuracy (LongBench avg)    Decode Speed    Memory
Full cache:     100% (baseline)             1.0×            1.0×
H2O:            89%  (drops retrieval)      ~2×             0.5×
SnapKV:         96%  (much better)          ~3.6×           0.5×
KIVI (2-bit):   97%  (best accuracy)        ~1.5×           0.38×
MiniKV:         98.5% (best overall)        ~2×             0.14×
```

**Why SnapKV beats H2O:**
- H2O uses global attention scores → misses per-head patterns
- SnapKV discovers that each attention head has its own "favorite" tokens
- SnapKV's observation window captures prompt-specific patterns before generation starts
- H2O has limited FlashAttention compatibility, SnapKV works with standard attention

**Why KIVI beats both on accuracy:**
- Eviction *permanently loses* information — dropped tokens are gone forever
- Quantization preserves all tokens, just at lower precision
- For tasks where every token matters (retrieval, reasoning), keeping all tokens at INT2 beats keeping 50% at FP16

**Why MiniKV beats everything:**
- Combines both: quantize to INT2 AND evict redundant tokens
- Layer-discriminative: not all layers need the same budget
- The "more tokens at lower precision" insight is the key — you get the coverage of keeping everything with the memory savings of aggressive compression

**Where eviction still wins:**
- If you need to run at very long contexts (100K+), even INT2 runs out of memory
- Eviction can achieve extreme compression (keep only 5% of tokens) that quantization can't match
- For streaming/infinite-length scenarios (StreamingLLM), eviction is the only option

---

## Implementation Ideas for Our Profiler

### What we can build to study these techniques:

#### 1. Token Importance Scorer (connects to H2O, SnapKV)

Add to pipeline_profiler.py's decode loop — extract attention weights and score each token:

```python
# In the decode loop, after each forward pass:
# model(..., output_attentions=True) returns attention weights

with torch.no_grad():
    outputs = model(
        input_ids=next_token,
        past_key_values=past_key_values,
        output_attentions=True,  # ← this is the key flag
        use_cache=True,
    )

# outputs.attentions is a tuple of (num_layers,) tensors
# Each tensor shape: (batch, num_heads, 1, seq_len)  (1 because we query with 1 token)
for layer_idx, attn_weights in enumerate(outputs.attentions):
    # attn_weights shape: (1, num_heads, 1, seq_len)
    # Sum across heads to get per-token importance for this layer
    token_scores = attn_weights[0, :, 0, :].sum(dim=0)  # shape: (seq_len,)
    
    # Track cumulative importance (H2O-style)
    cumulative_importance += token_scores
    
    # Track per-head importance (SnapKV-style)
    per_head_importance[layer_idx] += attn_weights[0, :, 0, :]
```

This gives us data to answer: "For this workload, how many tokens could be evicted with <1% quality loss?"

#### 2. KV Cache Size Tracker (connects to all papers)

Already partially built — extend to track per-step growth:

```python
# After each decode step:
kv_bytes = 0
for layer_kv in past_key_values:
    k, v = layer_kv[0], layer_kv[1]
    kv_bytes += k.numel() * k.element_size()
    kv_bytes += v.numel() * v.element_size()

step_metrics.append({
    "step": step,
    "kv_cache_bytes": kv_bytes,
    "kv_cache_mb": kv_bytes / (1024**2),
    "decode_latency_ms": step_latency,
    "seq_len": prompt_len + step,
})

# Plot: x=seq_len, y=decode_latency to find the "latency cliff"
# where KV cache starts competing with weights for bandwidth
```

#### 3. Bandwidth Split Analyzer (connects to ShadowKV, roofline analysis)

Use nsys kernel traces to measure actual bytes moved:

```python
# From our nsys profiler data, we already capture:
# - ampere_bf16_s16816gemm: weight × activation GEMMs (reads model weights)
# - flash_fwd_splitkv_kernel: attention computation (reads KV cache)
# - reshape_and_cache_flash_kernel: KV cache writes

# Compute bandwidth split at different sequence lengths:
weight_read_bytes = model_size_bytes  # constant per token
kv_read_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_bytes

bandwidth_ratio = kv_read_bytes / weight_read_bytes

# For Qwen2.5-7B at different sequence lengths:
# seq=64:   kv_read = 3.7 MB,  weight_read = 15.2 GB → ratio = 0.02%
# seq=2048: kv_read = 117 MB,  weight_read = 15.2 GB → ratio = 0.77%
# seq=8192: kv_read = 470 MB,  weight_read = 15.2 GB → ratio = 3.1%
# seq=32K:  kv_read = 1.88 GB, weight_read = 15.2 GB → ratio = 12.4%
# seq=128K: kv_read = 7.5 GB,  weight_read = 15.2 GB → ratio = 49.3%  ← KV starts mattering!
```

#### 4. Simulated Eviction Benchmark

Test what would happen if we applied eviction without actually implementing the full pipeline:

```python
# After running full inference (no eviction), analyze the attention patterns:
# 1. Collect all attention weights during decode
# 2. Simulate eviction at different budgets (keep 10%, 25%, 50%, 75%)
# 3. Compute "reconstruction error" — how different would the output be?

for budget in [0.10, 0.25, 0.50, 0.75]:
    tokens_to_keep = int(seq_len * budget)
    
    # H2O-style: keep top-K by cumulative attention + last W tokens
    h2o_kept = top_k_indices(cumulative_importance, tokens_to_keep)
    
    # SnapKV-style: keep per-head top-K from observation window
    snapkv_kept = per_head_top_k(per_head_importance, tokens_to_keep)
    
    # Report overlap between methods and potential quality impact
```

#### 5. Sequence Length Scaling Experiment

The most directly useful experiment — run pipeline profiler at increasing lengths:

```bash
# Run at multiple sequence lengths to find the latency cliff
for MAX_TOKENS in 16 64 256 512 1024 2048 4096; do
    python3 pipeline_profiler.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --max-tokens $MAX_TOKENS \
        --requests 3 \
        --kv-profile
done

# This produces data for:
# - Per-token latency vs sequence position (does it increase?)
# - KV cache memory growth curve
# - Total VRAM at each sequence length
# - Decode throughput degradation curve
```
