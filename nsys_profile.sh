#!/bin/bash
# ============================================================================
# Nsight Systems GPU Profiler Wrapper
# ============================================================================
# Profiles LLM inference at the GPU/driver level using nsys.
# Captures: CUDA kernel execution, memory transfers, NCCL comms,
#           CPU-GPU sync points, PCIe traffic, SM occupancy.
#
# Supports two backends:
#   1. vLLM  (Python, gives richest traces)
#   2. Ollama (Go/C++, GPU-level only but good for comparison)
#
# Usage:
#   cd ~/ollama/llm-infrence-benchmark-tests
#   source venv/bin/activate
#
#   # Profile vLLM inference
#   bash nsys_profile.sh vllm Qwen/Qwen2.5-7B-Instruct
#
#   # Profile Ollama inference
#   bash nsys_profile.sh ollama llama3
#
#   # Profile vLLM with multiple GPUs
#   bash nsys_profile.sh vllm Qwen/Qwen2.5-7B-Instruct --tp 4
#
#   # Profile with Nsight Compute (single-kernel deep dive)
#   bash nsys_profile.sh ncu vllm Qwen/Qwen2.5-7B-Instruct
#
# Options (env vars):
#   REQUESTS=3        Requests during profiling (default: 3, keep small)
#   MAX_TOKENS=128    Max output tokens (default: 128, keep small for ncu)
#   OUTPUT_DIR=...    Output directory
# ============================================================================

set -euo pipefail

REQUESTS=${REQUESTS:-3}
MAX_TOKENS=${MAX_TOKENS:-128}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-"./results/profiling/nsys_${TIMESTAMP}"}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_header() { echo -e "\n${CYAN}============================================================${NC}\n${CYAN}  $1${NC}\n${CYAN}============================================================${NC}"; }
log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: bash nsys_profile.sh <mode> <model> [options]"
    echo ""
    echo "Modes:"
    echo "  vllm <hf_model>     Profile vLLM offline inference"
    echo "  ollama <model_tag>  Profile Ollama inference"
    echo "  ncu <mode> <model>  Deep-dive single kernel with Nsight Compute"
    echo ""
    echo "Options:"
    echo "  --tp <N>            Tensor parallel degree (vLLM only, default: 1)"
    echo "  --hf-token <tok>    HuggingFace token for gated models"
    echo ""
    echo "Examples:"
    echo "  bash nsys_profile.sh vllm Qwen/Qwen2.5-7B-Instruct"
    echo "  bash nsys_profile.sh ollama mistral:7b"
    echo "  bash nsys_profile.sh ncu vllm Qwen/Qwen2.5-7B-Instruct"
    exit 1
}

[ $# -lt 2 ] && usage

MODE="$1"
shift

# Parse remaining args
MODEL=""
TP=1
HF_TOKEN=""
NCU_INNER_MODE=""

case "$MODE" in
    ncu)
        NCU_INNER_MODE="$1"
        shift
        ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp) TP="$2"; shift 2 ;;
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        *) [ -z "$MODEL" ] && MODEL="$1" || { log_error "Unknown arg: $1"; exit 1; }; shift ;;
    esac
done

[ -z "$MODEL" ] && { log_error "Model not specified"; usage; }

mkdir -p "$OUTPUT_DIR"

# ======================== VLLM PROFILING ========================
profile_vllm() {
    local model="$1"
    local model_safe=$(echo "$model" | tr '/' '_')
    local report_name="${OUTPUT_DIR}/nsys_vllm_${model_safe}"

    log_header "NSIGHT SYSTEMS — vLLM PROFILING"
    echo "  Model:     $model"
    echo "  TP:        $TP"
    echo "  Requests:  $REQUESTS"
    echo "  MaxTokens: $MAX_TOKENS"
    echo "  Output:    ${report_name}.nsys-rep"
    echo ""

    local hf_flag=""
    [ -n "$HF_TOKEN" ] && hf_flag="--hf-token $HF_TOKEN"

    log_info "Starting nsys profiling (this takes a while — model load + inference)..."

    nsys profile --trace-fork-before-exec=true \
        --output "$report_name" \
        --trace cuda,nvtx,cudnn,cublas,osrt \
        --cuda-memory-usage true \
        --gpuctxsw true \
        --force-overwrite true \
        --stats true \
        -- python3 vllm_profile.py \
            --model "$model" \
            --requests "$REQUESTS" \
            --warmup 1 \
             \
            --max-tokens "$MAX_TOKENS" \
            --tensor-parallel "$TP" \
            --output-dir "${OUTPUT_DIR}/vllm_metrics" \
             \
            $hf_flag

    log_success "nsys trace saved: ${report_name}.nsys-rep"

    # Generate summary stats
    log_info "Generating GPU stats summary..."
    nsys stats --report cuda_gpu_kern_sum \
        --format csv \
        --output "${report_name}_kernels" \
        "${report_name}.nsys-rep" 2>/dev/null || true

    nsys stats --report cuda_gpu_mem_size_sum \
        --format csv \
        --output "${report_name}_memops" \
        "${report_name}.nsys-rep" 2>/dev/null || true

    echo ""
    log_success "Profiling complete. Files:"
    echo "  Trace:       ${report_name}.nsys-rep"
    echo "  Kernel CSV:  ${report_name}_kernels.csv (if generated)"
    echo "  Memory CSV:  ${report_name}_memops.csv (if generated)"
    echo ""
    echo "  View with:   nsys-ui ${report_name}.nsys-rep"
    echo "  Or export:   nsys export --type sqlite ${report_name}.nsys-rep"
}

# ======================== OLLAMA PROFILING ========================
profile_ollama() {
    local model="$1"
    local model_safe=$(echo "$model" | tr ':/' '_')
    local report_name="${OUTPUT_DIR}/nsys_ollama_${model_safe}"

    log_header "NSIGHT SYSTEMS — OLLAMA PROFILING"
    echo "  Model:     $model"
    echo "  Requests:  $REQUESTS"
    echo "  MaxTokens: $MAX_TOKENS"
    echo "  Output:    ${report_name}.nsys-rep"
    echo ""

    # Check Ollama is running
    if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
        log_error "Ollama not running. Start it first: ollama serve"
        exit 1
    fi

    # Warmup
    log_info "Warming up ${model}..."
    curl -s http://localhost:11434/api/generate \
        -d "{\"model\": \"${model}\", \"prompt\": \"Hello\", \"stream\": false}" \
        > /dev/null 2>&1
    sleep 2

    # Create a small benchmark script to profile
    local bench_script="${OUTPUT_DIR}/ollama_bench_for_nsys.py"
    cat > "$bench_script" << 'PYEOF'
import json
import sys
import time
import requests

model = sys.argv[1]
num_requests = int(sys.argv[2])
max_tokens = int(sys.argv[3])

prompts = [
    "Explain the concept of recursion in programming in 3 sentences.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis in simple terms.",
]

for i in range(num_requests):
    prompt = prompts[i % len(prompts)]
    t0 = time.monotonic()
    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens}
    })
    t1 = time.monotonic()
    data = resp.json()
    tokens = data.get("eval_count", 0)
    print(f"Request {i}: {tokens} tokens in {t1-t0:.2f}s ({tokens/(t1-t0):.1f} tok/s)")
PYEOF

    log_info "Starting nsys profiling of Ollama inference..."

    # Profile the client requests (GPU activity is captured from the Ollama server process)
    # We use --trace cuda to capture all GPU activity system-wide
    nsys profile --trace-fork-before-exec=true \
        --output "$report_name" \
        --trace cuda,nvtx,cudnn,cublas \
        --cuda-memory-usage true \
        --sample none \
        --force-overwrite true \
        --stats true \
        -- python3 "$bench_script" "$model" "$REQUESTS" "$MAX_TOKENS"

    log_success "nsys trace saved: ${report_name}.nsys-rep"

    # Generate stats
    nsys stats --report cuda_gpu_kern_sum \
        --format csv \
        --output "${report_name}_kernels" \
        "${report_name}.nsys-rep" 2>/dev/null || true

    echo ""
    log_success "Profiling complete. Files:"
    echo "  Trace:       ${report_name}.nsys-rep"
    echo "  Kernel CSV:  ${report_name}_kernels.csv (if generated)"
    echo ""
    echo "  View with:   nsys-ui ${report_name}.nsys-rep"
}

# ======================== NCU DEEP DIVE ========================
profile_ncu() {
    local inner_mode="$1"
    local model="$2"
    local model_safe=$(echo "$model" | tr '/:' '_')
    local report_name="${OUTPUT_DIR}/ncu_${inner_mode}_${model_safe}"

    log_header "NSIGHT COMPUTE — KERNEL DEEP DIVE"
    echo "  Mode:      $inner_mode"
    echo "  Model:     $model"
    echo "  Output:    ${report_name}.ncu-rep"
    echo ""
    log_warn "ncu profiles individual kernel launches. This is VERY slow."
    log_warn "Using --launch-count 10 to limit to first 10 kernel launches."
    log_warn "Use REQUESTS=1 MAX_TOKENS=32 for fast results."
    echo ""

    if [ "$inner_mode" = "vllm" ]; then
        local hf_flag=""
        [ -n "$HF_TOKEN" ] && hf_flag="--hf-token $HF_TOKEN"

        ncu --target-processes all \
            --launch-count 10 \
            --set full \
            --output "$report_name" \
            --force-overwrite \
            -- python3 vllm_profile.py \
                --model "$model" \
                --requests 1 \
                --warmup 0 \
                 \
                --max-tokens 32 \
                --tensor-parallel "$TP" \
                --output-dir "${OUTPUT_DIR}/ncu_metrics" \
                 \
                $hf_flag
    else
        log_error "ncu for Ollama requires profiling the ollama process directly."
        log_info "Use: ncu --target-processes all --launch-count 10 --set full -o $report_name -- ollama run $model 'Hello'"
        exit 1
    fi

    log_success "ncu report saved: ${report_name}.ncu-rep"
    echo "  View with: ncu-ui ${report_name}.ncu-rep"
}

# ======================== DISPATCH ========================
case "$MODE" in
    vllm)
        profile_vllm "$MODEL"
        ;;
    ollama)
        profile_ollama "$MODEL"
        ;;
    ncu)
        profile_ncu "$NCU_INNER_MODE" "$MODEL"
        ;;
    *)
        log_error "Unknown mode: $MODE"
        usage
        ;;
esac

echo ""
log_header "NEXT STEPS"
echo "  1. Copy .nsys-rep / .ncu-rep to local machine:"
echo "     scp jbostok@cci-aries:~/ollama/llm-infrence-benchmark-tests/${OUTPUT_DIR}/*.nsys-rep ~/Desktop/"
echo ""
echo "  2. Open in Nsight Systems UI (nsys-ui) or Nsight Compute UI (ncu-ui)"
echo ""
echo "  3. Or export to SQLite for custom analysis:"
echo "     nsys export --type sqlite <file>.nsys-rep"
echo ""
echo "  4. Compare vLLM vs Ollama kernel profiles:"
echo "     diff <(sort ${OUTPUT_DIR}/*_vllm_*_kernels.csv) <(sort ${OUTPUT_DIR}/*_ollama_*_kernels.csv)"
echo ""
