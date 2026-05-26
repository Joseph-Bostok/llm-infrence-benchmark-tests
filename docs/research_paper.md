# Profiling-Guided KV Cache Optimization: Token Eviction and Semantic Chunking for Long-Context LLM Inference

**Joseph Bostok**  
Department of Computer Science, University of North Carolina at Charlotte  

---

## Abstract

The key-value (KV) cache is the central scalability bottleneck in long-context large language model (LLM) inference. During autoregressive generation, each decode step reads the entire cached key-value state — a cost that grows linearly with sequence length and, for real-world workloads such as multi-turn dialogue, long-document question answering, and retrieval-augmented generation (RAG), can reach tens of gigabytes of GPU memory. At 32K tokens, the KV cache for a 7-billion-parameter model consumes 1.9 GB and accounts for over 10% of total memory bandwidth; at 128K tokens, it approaches parity with the model weights themselves. This growth forces a fundamental trade-off: retaining all cached tokens guarantees output quality but limits batch size and throughput, while aggressive cache reduction risks degrading accuracy on retrieval-sensitive and reasoning-intensive tasks.

Two families of KV cache optimization have emerged to address this problem. **Token eviction** methods (H2O, SnapKV) score individual tokens by cumulative attention weight and discard the lowest-scoring entries, achieving up to 10× memory compression but fragmenting semantic context and struggling on tasks that require retrieving specific details from the distant past. **Semantic chunking** methods (ChunkKV) group tokens into contiguous segments and evict at the chunk level, preserving linguistic coherence and outperforming per-token eviction by up to 8.7% on long-context benchmarks, but operating with coarser granularity that may over-retain irrelevant passages. Both approaches are typically evaluated on standard benchmarks — LongBench (single/multi-document QA, summarization, few-shot learning, code completion, synthetic retrieval) and Needle-in-a-Haystack (NIAH) — yet the choice between eviction strategies remains largely heuristic, with no empirical tooling to connect cache behavior to downstream quality for a given workload.

We present **PInsight**, a profiling-guided framework for KV cache optimization that provides per-token, per-layer visibility into cache behavior during inference. PInsight instruments the decode loop of HuggingFace Transformer models to capture three data streams simultaneously: (1) per-step KV cache memory growth and decode latency, (2) per-layer attention weight distributions for token importance scoring, and (3) GPU kernel traces via NVTX-annotated Nsight Systems for bandwidth attribution. Using real attention data from Qwen2.5-7B-Instruct on NVIDIA A100 GPUs, we establish that attention is far more concentrated than previously reported — the first four tokens absorb **40.8%** of cumulative attention (the "attention sink" effect), and only **0.5%** of tokens qualify as heavy hitters, compared to H2O's original estimate of ~5%. Our eviction simulation shows that at a 50% cache budget, H2O-style eviction retains 82% of attention mass but may miss semantically coherent context that chunk-level eviction preserves. We further identify the **bandwidth crossover point** at ~32K tokens, where KV cache reads surpass 10% of total HBM bandwidth and cache optimization begins yielding measurable throughput gains. These findings provide practitioners with empirical, workload-specific guidance for choosing between token-level and chunk-level eviction — a decision that existing benchmarks evaluate only post-hoc.

---

## 1. Introduction

### 1.1 The KV Cache Problem

The deployment of large language models in production systems has exposed a fundamental tension between the autoregressive generation mechanism and the memory-bandwidth constraints of modern GPU hardware. During decode, each new token requires a full forward pass through the model — reading billions of weight parameters from GPU high-bandwidth memory (HBM) to compute a single output. On an NVIDIA A100, single-token decode operates at an arithmetic intensity of approximately 1 FLOP/byte, a factor of 208× below the GPU's compute ridge point [1]. The decode phase is firmly memory-bandwidth limited.

The key-value cache addresses the computational cost of this bottleneck by caching attention keys and values from previous tokens, converting an O(n²) recomputation into an O(n) memory read. But this creates a second memory problem: the KV cache grows linearly with sequence length and must be read *in its entirety* at every decode step. For Qwen2.5-7B with grouped-query attention (GQA), the cache requires approximately 57 KB per token:

| Sequence Length | KV Cache Size | % of Weight Bandwidth |
|:-:|:-:|:-:|
| 64 tokens | 3.7 MB | 0.02% |
| 2,048 tokens | 117 MB | 0.8% |
| 8,192 tokens | 470 MB | 3.1% |
| 32,768 tokens | 1.9 GB | 12.3% |
| 131,072 tokens | 7.5 GB | 49.3% |

At short contexts, the KV cache is negligible. At long contexts — well within the advertised window of production models — it becomes a dominant factor in both latency and memory.

### 1.2 Real-World Workloads That Hit This Wall

The KV cache scaling problem is not hypothetical. Three classes of production workloads routinely push models into the regime where cache management determines system viability:

**Multi-turn dialogue and AI agents.** Conversational AI systems and autonomous agents (e.g., tool-using LLMs, code assistants) maintain full conversation histories across many turns. Each user interaction appends to the KV cache without bound. A 50-turn conversation with 200 tokens per turn reaches 10K context tokens; an agentic workflow with tool call/response cycles can exceed 50K tokens within a single session. Without cache management, GPU memory fills within minutes, forcing either context truncation (quality loss) or request preemption (latency spikes).

**Long-document question answering and RAG.** Retrieval-augmented generation systems inject retrieved passages — often 5–20 documents of 500–2,000 tokens each — into the prompt before generating an answer. A RAG system with 10 retrieved passages at 1,000 tokens each requires 10K tokens of KV cache before generation even begins. Legal document analysis, medical records review, and codebase understanding routinely process 16K–128K token contexts, placing these workloads squarely in the bandwidth crossover zone.

**Code generation.** Generating or completing code requires maintaining deep logical consistency over long sequences — variable definitions, function signatures, and architectural constraints established hundreds or thousands of tokens earlier must remain available in the cache. Aggressive eviction risks dropping foundational definitions that later tokens depend on, leading to semantically broken output. Repository-level code completion (as evaluated in LongBench) represents one of the most cache-sensitive workloads.

### 1.3 Token Eviction vs. Semantic Chunking

The optimization community has responded with two complementary approaches to reduce KV cache memory, each with distinct strengths and failure modes.

**Token-level eviction** methods assign an importance score to each cached token and discard the lowest-scoring entries. H2O [2] tracks cumulative attention — the total attention weight a token has received across all layers and decode steps — and retains the top-K "heavy hitters" plus a sliding window of recent tokens. SnapKV [3] improves on this by using a per-head observation window: the last few prompt tokens reveal which cached positions each attention head consistently focuses on, enabling head-specific selection. Both achieve significant cache compression — SnapKV reports 3.6× decode speedup and 8.2× memory reduction at 16K context — with minimal accuracy loss on most LongBench tasks.

But token-level eviction has a structural weakness: **it fragments semantic context.** When individual tokens are scattered across the sequence, the model loses the contiguous context around each retained token. If token 45 is kept but tokens 43, 44, 46, and 47 are evicted, the model retains the word at position 45 without the sentence it belongs to. This fragmentation is particularly harmful for tasks requiring synthesis across related passages (multi-document QA) or understanding of logical structure (code completion).

**Semantic chunking** (ChunkKV [14]) addresses this by grouping tokens into fixed-size contiguous chunks (e.g., 10–16 tokens) and evaluating importance at the chunk level. Entire chunks are kept or discarded, preserving the linguistic unit around each retained token. ChunkKV additionally exploits **layer-wise index reuse** — the observation that adjacent transformer layers select nearly identical chunks — to amortize the cost of importance computation across layers, achieving 26.5% higher throughput than full-cache baselines.

On LongBench, ChunkKV outperforms H2O and SnapKV by up to 8.7% at the same compression ratio, with the largest gains on summarization and multi-document QA tasks where cross-sentence coherence matters most. However, chunk-level eviction has its own limitation: **granularity is coarser**, meaning that an entire chunk may be retained because one constituent token scores highly, even if the rest of the chunk is irrelevant.

### 1.4 The Benchmark Landscape

Both eviction and chunking methods are evaluated on a common set of benchmarks:

| Benchmark | What It Tests | Tasks | Length Range |
|-----------|-------------|:-----:|:-----------:|
| **LongBench** | Broad long-context understanding | 21 datasets across 6 categories: single-doc QA, multi-doc QA, summarization, few-shot, synthetic, code | 4K – 16K+ tokens |
| **Needle-in-a-Haystack (NIAH)** | Exact retrieval from long context | Single fact buried in irrelevant padding | 4K – 128K+ tokens |
| **RULER** | Synthetic long-range tasks | Multi-key retrieval, variable tracking, needle variants | 4K – 128K tokens |
| **GSM8K (many-shot)** | In-context reasoning with many examples | Math word problems with 32–128 few-shot examples | 8K – 32K tokens |

These benchmarks measure the *accuracy* impact of cache compression after the fact — run inference with a reduced cache and check whether the output quality degrades. What they do not measure is *why* a particular eviction strategy succeeds or fails on a given task: which tokens were evicted, which layers lost the most information, and how cache pressure affected decode latency.

### 1.5 Our Approach: Profiling-Guided Optimization

We argue that the choice between token-level eviction and semantic chunking should not be made through benchmark trial-and-error but through **profiling-guided analysis** of the specific workload's attention patterns, cache pressure profile, and bandwidth utilization.

PInsight provides the observability layer that connects cache behavior to downstream quality. During inference, it captures:

1. **Token importance scores** — per-layer, per-head cumulative attention weights (H2O-style) and observation-window distributions (SnapKV-style), enabling direct comparison of eviction strategies on the same workload.
2. **KV cache growth curves** — measured and theoretical cache size at every decode step, revealing where cache pressure begins affecting latency.
3. **Attention sink analysis** — quantification of the attention mass absorbed by initial tokens (StreamingLLM [8] pattern), which must be preserved regardless of eviction strategy.
4. **Bandwidth crossover detection** — identification of the sequence length at which KV cache reads surpass a configurable fraction of total HBM bandwidth (default 10%), marking the point where cache optimization yields measurable throughput improvement.
5. **Eviction simulation** — evaluation of token-level vs. chunk-level eviction at multiple budgets using real attention data, providing attention coverage and memory savings estimates before any eviction policy is deployed.

### 1.6 How Our Approach Differs from Benchmark-Based Evaluation

Existing work evaluates eviction and chunking strategies by running LongBench/NIAH/RULER with a compressed cache and measuring the accuracy delta. This tells you *what happened* but not *why*. Our profiling-guided approach differs in three ways:

| Dimension | Benchmark Evaluation | PInsight (Profiling-Guided) |
|-----------|:-------------------:|:--------------------------:|
| **When** | Post-hoc (compress first, evaluate after) | Proactive (analyze attention, then decide strategy) |
| **Granularity** | Task-level accuracy (single score per benchmark) | Per-token, per-layer, per-head importance maps |
| **Output** | "This method scored X on LongBench" | "At 50% budget, eviction captures 82% of attention mass; tokens 0, 19, 106. 81, 82 are critical; layer 4 is most selective" |
| **Actionability** | Try a different method if accuracy drops | Choose between eviction/chunking based on attention distribution shape |
| **Workload specificity** | Same benchmark for all workloads | Profiles *your* prompt, *your* model, *your* sequence length |

### 1.7 Key Findings

Through experiments on Qwen2.5-7B-Instruct (28 layers, 4 KV heads, GQA 7:1) running on NVIDIA A100-40GB GPUs, we establish:

1. **Attention is far more concentrated than prior work suggests.** The first four tokens absorb 40.8% of all cumulative attention — nearly three times the ~15% predicted by StreamingLLM analysis. Only 0.5% of tokens are heavy hitters, compared to H2O's estimate of ~5%. This concentration implies that eviction is viable at much higher compression ratios than commonly assumed.

2. **Token importance is extremely spiky.** The importance distribution is dominated by a few positions (tokens 0, 19, 106, 81, 82, 144, 131, 77, 87, 157 in our Qwen2.5-7B run), with most tokens near zero. This favors token-level eviction over chunking for workloads where the important tokens are scattered across the sequence.

3. **Layer selectivity varies dramatically.** Layer 4 is the most selective (lowest attention entropy) while Layer 1 is the most diffuse. This validates MiniKV's layer-discriminative insight: early layers require larger KV budgets, while late layers can be compressed more aggressively.

4. **The bandwidth crossover occurs at ~32K tokens.** Below this threshold, KV cache reads are negligible relative to weight reads. Above it, cache optimization translates directly to throughput improvement. This finding defines the operational boundary above which eviction and chunking strategies deliver real performance gains.

5. **Decode latency grows measurably even at short sequences.** A 9% increase in per-token latency over just 128 generated tokens demonstrates that cache pressure is detectable from the first hundred tokens — well before the 32K bandwidth crossover.

6. **A 50% eviction budget retains 82% of attention mass.** Using real attention data, our H2O-style simulation shows that half the cache can be discarded while preserving the vast majority of information the model actually attends to. At 75% budget, coverage reaches 93.6%; at 90%, it reaches 98.2%.

### 1.8 Paper Organization

The remainder of this paper is organized as follows. Section 2 surveys related work across token-level eviction, semantic chunking, and the benchmark landscape. Section 3 describes the PInsight architecture. Section 4 details our experimental setup. Section 5 presents findings on attention patterns, cache pressure, and eviction simulation. Section 6 compares profiling-guided optimization with benchmark-based evaluation on real-world workload categories. Section 7 concludes with future work directions including integration with the NVIDIA KVPress framework and extension to reasoning-model workloads (R-KV).

---

## References

[1] Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4), 65-76.

[2] Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., Ré, C., Barrett, C., & Wang, Z. (2023). H2O: Heavy-hitter oracle for efficient generative inference of large language models. *NeurIPS 2023*. arXiv:2306.14048.

[3] Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A., Ye, H., Cai, T., McAuley, J., & Huang, G. (2024). SnapKV: LLM knows what you are looking for before generation. arXiv:2404.14469.

[4] Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V., Chen, B., & Hu, X. (2024). KIVI: A tuning-free asymmetric 2bit quantization for KV cache. *ICML 2024*. arXiv:2402.02750.

[5] Zhong, K., Yue, Y., Shi, B., Li, Y., Zhang, X., Zhao, R., & Yin, H. (2024). MiniKV: Pushing the limits of LLM inference via 2-bit layer-discriminative KV cache. arXiv:2411.18077.

[6] Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023). Efficient memory management for large language model serving with PagedAttention. *SOSP 2023*. arXiv:2309.06180.

[7] Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training generalized multi-query transformer models from multi-head checkpoints. arXiv:2305.13245.

[8] Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). Efficient streaming language models with attention sinks. arXiv:2309.17453.

[9] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS 2022*. arXiv:2205.14135.

[10] Sun, Y., et al. (2024). ShadowKV: KV cache in shadows for high-throughput long-context LLM inference. arXiv:2410.21465.

[11] Bai, Y., et al. (2023). LongBench: A bilingual, multitask benchmark for long context understanding. *ACL 2024*. arXiv:2308.14508.

[12] Hsieh, C.-Y., et al. (2024). RULER: What's the real context size of your long-context language models? arXiv:2404.06654.

[13] Kamradt, G. (2023). Needle in a Haystack: Pressure testing LLMs. GitHub.

[14] Xu, Y., et al. (2025). ChunkKV: Semantic-preserving KV cache compression for efficient long-context LLM inference. *NeurIPS 2025*. arXiv:2502.00299.

[15] Zhu, Y., et al. (2025). LMCache: Sharing across users, instances, and models with an efficient KV cache layer. *2025*.

[16] Qin, R., et al. (2024). Mooncake: A KVCache-centric disaggregated architecture for LLM serving. arXiv:2407.00079.
