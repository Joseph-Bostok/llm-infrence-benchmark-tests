#!/bin/bash
# ============================================================================
# Tier 1 Multi-Model Benchmark Suite (7-8B Parameters)
# ============================================================================
# Runs ollama_benchmark.py across all Tier 1 models with:
#   - Automatic model pulling for models not yet on the server
#   - Warmup request before each benchmark (avoids cold-start skew)
#   - Resumable: skips models that already have results
#   - Per-model result directories
#   - Summary comparison at the end
# ============================================================================
# Usage:
#   cd ~/ollama/llm-infrence-benchmark-tests
#   source venv/bin/activate
#   bash tier1_benchmark.sh
#
# Options:
#   REQUESTS=25       Number of benchmark requests (default: 25)
#   FORCE=true        Re-run even if results exist (default: false)
#   SKIP_PULL=true    Don't pull missing models (default: false)
#   VERBOSE=true      Show per-token output (default: false)
# ============================================================================

set -euo pipefail

# --- Configuration ---
TIER="tier1_7-8B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="./results/${TIER}"
REQUESTS=${REQUESTS:-25}
FORCE=${FORCE:-false}
SKIP_PULL=${SKIP_PULL:-false}
VERBOSE=${VERBOSE:-false}
OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}

# Tier 1 models: 7-8B parameter range
# Format: "ollama_name:tag|display_name|param_count|architecture"
MODELS=(
    "llama3:latest|Llama3-8B|8B|Dense-Transformer"
    "llama3.1:latest|Llama3.1-8B|8B|Dense-Transformer"
    "deepseek-r1:latest|DeepSeek-R1-7B|7B|Dense-Transformer-CoT"
    "qwen2.5:7b|Qwen2.5-7B|7B|Dense-Transformer"
    "gemma2:9b|Gemma2-9B|9B|Dense-Transformer"
    "mistral:7b|Mistral-7B|7B|Dense-Transformer-SWA"
)

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- Helper Functions ---
log_header() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}============================================================${NC}"
}

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

check_ollama() {
    if ! curl -s "${OLLAMA_HOST}" > /dev/null 2>&1; then
        log_error "Ollama server not reachable at ${OLLAMA_HOST}"
        log_info "Start it with: ollama serve"
        exit 1
    fi
    log_success "Ollama server is running at ${OLLAMA_HOST}"
}

is_model_available() {
    local model_name="$1"
    ollama list 2>/dev/null | grep -q "^${model_name}" && return 0 || return 1
}

warmup_model() {
    local model_name="$1"
    log_info "Warming up ${model_name} (loading into GPU memory)..."
    local start_time=$(date +%s%N)
    curl -s "${OLLAMA_HOST}/api/generate" \
        -d "{\"model\": \"${model_name}\", \"prompt\": \"Hello\", \"stream\": false}" \
        > /dev/null 2>&1
    local end_time=$(date +%s%N)
    local elapsed=$(( (end_time - start_time) / 1000000 ))
    log_success "Warmup complete (${elapsed}ms)"
}

# --- Main Script ---
log_header "TIER 1 MULTI-MODEL BENCHMARK SUITE (7-8B)"
echo ""
echo "  Timestamp:    ${TIMESTAMP}"
echo "  Results Dir:  ${RESULTS_BASE}/"
echo "  Requests:     ${REQUESTS} per model"
echo "  Models:       ${#MODELS[@]}"
echo "  Force Rerun:  ${FORCE}"
echo ""

# Check prerequisites
check_ollama

if [ ! -f "scripts/ollama_benchmark.py" ]; then
    log_error "ollama_benchmark.py not found. Are you in ~/ollama/llm-infrence-benchmark-tests/ ?"
    exit 1
fi
log_success "Benchmark script found"

# Create results directory
mkdir -p "${RESULTS_BASE}"

# --- Phase 1: Pull Missing Models ---
log_header "PHASE 1: MODEL AVAILABILITY CHECK"

AVAILABLE_MODELS=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_tag display_name param_count arch <<< "$entry"

    if is_model_available "$model_tag"; then
        log_success "${display_name} (${model_tag}) — available"
        AVAILABLE_MODELS+=("$entry")
    elif [ "$SKIP_PULL" = "true" ]; then
        log_warn "${display_name} (${model_tag}) — not available, skipping (SKIP_PULL=true)"
    else
        log_info "Pulling ${display_name} (${model_tag})..."
        if ollama pull "${model_tag}"; then
            log_success "${display_name} pulled successfully"
            AVAILABLE_MODELS+=("$entry")
        else
            log_error "Failed to pull ${display_name} — skipping"
        fi
    fi
done

echo ""
log_info "Models ready: ${#AVAILABLE_MODELS[@]} / ${#MODELS[@]}"

if [ ${#AVAILABLE_MODELS[@]} -eq 0 ]; then
    log_error "No models available. Exiting."
    exit 1
fi

# --- Phase 2: Run Benchmarks ---
log_header "PHASE 2: RUNNING BENCHMARKS"

COMPLETED=0
SKIPPED=0
FAILED=0

# Build verbose flag
VFLAG=""
if [ "$VERBOSE" = "true" ]; then
    VFLAG="-v"
fi

for entry in "${AVAILABLE_MODELS[@]}"; do
    IFS='|' read -r model_tag display_name param_count arch <<< "$entry"

    # Sanitize model name for directory (replace : and / with _)
    dir_name=$(echo "$display_name" | tr ' /:' '_')
    model_dir="${RESULTS_BASE}/${dir_name}"
    result_file="${model_dir}/results.json"

    echo ""
    log_header "BENCHMARKING: ${display_name} (${param_count}, ${arch})"

    # Check if results already exist
    if [ -f "$result_file" ] && [ "$FORCE" != "true" ]; then
        log_warn "Results already exist at ${result_file}"
        log_warn "Skipping (use FORCE=true to re-run)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    mkdir -p "$model_dir"

    # Warmup: load model into GPU memory
    warmup_model "$model_tag"

    # Small delay to let GPU state stabilize
    sleep 2

    # Run benchmark
    log_info "Running ${REQUESTS} requests against ${model_tag}..."
    local_start=$(date +%s)

    if python3 scripts/ollama_benchmark.py \
        --model "$model_tag" \
        -o "$result_file" \
        -t "${model_dir}/token_log.txt" \
        $VFLAG \
        2>&1 | tee "${model_dir}/benchmark_stdout.log"; then

        local_end=$(date +%s)
        elapsed=$((local_end - local_start))
        log_success "${display_name} completed in ${elapsed}s"
        log_success "Results: ${result_file}"

        # Save metadata
        cat > "${model_dir}/metadata.json" << METAEOF
{
    "model_tag": "${model_tag}",
    "display_name": "${display_name}",
    "param_count": "${param_count}",
    "architecture": "${arch}",
    "tier": "${TIER}",
    "requests": ${REQUESTS},
    "timestamp": "${TIMESTAMP}",
    "elapsed_seconds": ${elapsed},
    "ollama_host": "${OLLAMA_HOST}"
}
METAEOF

        COMPLETED=$((COMPLETED + 1))
    else
        log_error "${display_name} benchmark FAILED"
        FAILED=$((FAILED + 1))
    fi

    # Unload model to free GPU memory before next model
    log_info "Unloading ${model_tag} from GPU..."
    curl -s "${OLLAMA_HOST}/api/generate" \
        -d "{\"model\": \"${model_tag}\", \"keep_alive\": 0}" \
        > /dev/null 2>&1 || true
    sleep 3
done

# --- Phase 3: Summary ---
log_header "BENCHMARK COMPLETE"
echo ""
echo "  Completed:  ${COMPLETED}"
echo "  Skipped:    ${SKIPPED}"
echo "  Failed:     ${FAILED}"
echo "  Results in: ${RESULTS_BASE}/"
echo ""

# List all result files
log_info "Result files:"
for entry in "${AVAILABLE_MODELS[@]}"; do
    IFS='|' read -r model_tag display_name param_count arch <<< "$entry"
    dir_name=$(echo "$display_name" | tr ' /:' '_')
    result_file="${RESULTS_BASE}/${dir_name}/results.json"
    if [ -f "$result_file" ]; then
        size=$(du -h "$result_file" | cut -f1)
        echo -e "  ${GREEN}✓${NC} ${display_name}: ${result_file} (${size})"
    else
        echo -e "  ${RED}✗${NC} ${display_name}: no results"
    fi
done

echo ""
log_info "Next step: run the comparison script"
echo "  python3 tier1_compare.py ${RESULTS_BASE}/"
echo ""
