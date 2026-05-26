# PInsight Server Deployment Guide

Step-by-step instructions for deploying PInsight on your GPU server (H100/H200).

---

## Prerequisites

- NVIDIA GPU (H100 or H200 recommended, A100 also works)
- CUDA 12.x installed
- Python 3.10+
- ~50GB disk space for models

---

## Step 1: Clone the Repository

```bash
git clone <your-repo-url> llm-inference-benchmark-tests
cd llm-inference-benchmark-tests
```

## Step 2: Create Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

## Step 3: Install Dependencies

```bash
# Core dependencies (always needed)
pip install requests matplotlib numpy

# vLLM (requires CUDA GPU)
pip install vllm

# GuideLLM (for load testing evaluation)
pip install guidellm

# Optional: for Nsight Systems integration
# (nsys is installed separately via NVIDIA CUDA Toolkit)
```

## Step 4: Download the Model

```bash
# Option A: vLLM will auto-download on first run
# Option B: Pre-download with huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

## Step 5: Verify Setup

```bash
# Test workload classifier (no GPU needed)
python3 workload_classifier.py

# Test tuning engine (no GPU needed)
python3 tuning_engine.py --compare-all

# Test full pipeline in dry-run mode (no GPU needed)
python3 run_experiments.py --dry-run

# Generate demo figures (no GPU needed)
python3 visualizations/paper_figures.py --demo
```

If all four commands pass, you're ready for GPU experiments.

---

## Step 6: Run Baseline Experiments (GPU Required)

### Option A: Automated (Recommended)

```bash
# Runs the full pipeline: baseline → profile → tune → compare
python3 run_experiments.py --model Qwen/Qwen2.5-7B-Instruct --phase all
```

### Option B: Manual (Step by Step)

```bash
# 1. Start vLLM server
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.90 &

# Wait for server to be ready (check logs)
sleep 60

# 2. Run profiler on each workload
python3 vllm_kv_profiler.py \
    --server http://localhost:8000 \
    --workload workloads/dialogue.json \
    --output results/baseline_dialogue.json

python3 vllm_kv_profiler.py \
    --server http://localhost:8000 \
    --workload workloads/rag.json \
    --output results/baseline_rag.json

python3 vllm_kv_profiler.py \
    --server http://localhost:8000 \
    --workload workloads/code.json \
    --output results/baseline_code.json

python3 vllm_kv_profiler.py \
    --server http://localhost:8000 \
    --workload workloads/reasoning.json \
    --output results/baseline_reasoning.json

# 3. Stop server
kill %1

# 4. Get tuned config for each workload
python3 tuning_engine.py --workload dialogue --model Qwen/Qwen2.5-7B-Instruct
python3 tuning_engine.py --workload rag --model Qwen/Qwen2.5-7B-Instruct
python3 tuning_engine.py --workload code --model Qwen/Qwen2.5-7B-Instruct
python3 tuning_engine.py --workload reasoning --model Qwen/Qwen2.5-7B-Instruct

# 5. Re-run with tuned configs (example for dialogue)
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 4096 \
    --enable-prefix-caching \
    --max-model-len 8192 &

sleep 60

python3 vllm_kv_profiler.py \
    --server http://localhost:8000 \
    --workload workloads/dialogue.json \
    --output results/tuned_dialogue.json

kill %1
# Repeat for rag, code, reasoning with their respective configs
```

## Step 7: Generate Figures

```bash
python3 visualizations/paper_figures.py --results results/experiments/
```

Figures will be saved to `visualizations/figures/`.

---

## Step 8: Run GuideLLM Evaluation (Optional)

```bash
# Start vLLM server first (baseline config)
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct --port 8000 &

# Run GuideLLM sweep profile
guidellm --target http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --rate-type sweep \
    --max-seconds 120 \
    --output results/guidellm_baseline.json
```

---

## File Map

```
llm-inference-benchmark-tests/
├── workload_classifier.py      # Classifies prompts → workload type
├── tuning_engine.py            # Workload type → optimal vLLM config
├── vllm_kv_profiler.py         # Profiles running vLLM server
├── run_experiments.py          # Orchestrates full experiment pipeline
├── workloads/                  # Curated prompt datasets
│   ├── dialogue.json
│   ├── rag.json
│   ├── code.json
│   └── reasoning.json
├── visualizations/
│   └── paper_figures.py        # Generates publication figures
├── results/                    # Experiment output (gitignored)
│   └── experiments/
│       ├── baseline_*.json
│       ├── tuned_*.json
│       └── comparison.json
├── DEPLOYMENT.md               # This file
│
│ --- Existing tools (from earlier work) ---
├── pipeline_profiler.py        # HuggingFace pipeline stage profiler
├── kv_cache_profiler.py        # Attention-based KV cache analysis
├── nsys_profiler.py            # Nsight Systems wrapper
├── kineto_profiler.py          # PyTorch Kineto wrapper
└── scripts/
    ├── pinsight_correlator.py  # Cross-layer trace correlation
    └── guidellm_benchmark.py   # GuideLLM integration
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `--gpu-memory-utilization` to 0.80
- Use a smaller model first (e.g., Qwen2.5-1.5B-Instruct)
- Reduce `--max-model-len`

### vLLM server won't start
- Check CUDA version: `nvcc --version` (need 12.x)
- Check GPU is visible: `nvidia-smi`
- Try: `CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server ...`

### "requests" import error
- `pip install requests`

### Figures look wrong
- `pip install matplotlib numpy`
- Check that results JSON files exist in the results directory
