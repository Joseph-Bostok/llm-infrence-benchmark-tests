#!/bin/bash
# LLM Inference Benchmark Runner
# Runs Ollama, Etalon, GuideLLM, and Reasoning benchmarks
# Supports multimodal analysis and PInsight trace correlation

set -e

# Add local Python bin to PATH for GuideLLM
export PATH="$PATH:$HOME/Library/Python/3.9/bin"

# Configuration
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODEL="${MODEL:-llama2}"
OUTPUT_DIR="./results/$(date +%Y%m%d_%H%M%S)"
NUM_REQUESTS="${NUM_REQUESTS:-5}"
ENABLE_GUIDELLM="${ENABLE_GUIDELLM:-true}"
ENABLE_REASONING="${ENABLE_REASONING:-true}"
ENABLE_ETALON="${ENABLE_ETALON:-false}"
ENABLE_PINSIGHT="${ENABLE_PINSIGHT:-false}"
LTTNG_TRACE_DIR="${LTTNG_TRACE_DIR:-}"
NSIGHT_TRACE="${NSIGHT_TRACE:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}LLM Inference Benchmark Suite v2.0${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Configuration:"
echo "  Ollama Host:     $OLLAMA_HOST"
echo "  Model:           $MODEL"
echo "  Output Dir:      $OUTPUT_DIR"
echo "  Requests:        $NUM_REQUESTS"
echo "  GuideLLM:        $ENABLE_GUIDELLM"
echo "  Reasoning:       $ENABLE_REASONING"
echo "  Etalon:          $ENABLE_ETALON"
echo "  PInsight:        $ENABLE_PINSIGHT"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/ollama"
mkdir -p "$OUTPUT_DIR/etalon"
mkdir -p "$OUTPUT_DIR/guidellm"
mkdir -p "$OUTPUT_DIR/reasoning"
mkdir -p "./visualizations"

# ============================================
# Phase 1: Check Prerequisites
# ============================================
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if Ollama server is running
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ Ollama server is running${NC}"
else
    echo -e "${RED}  ✗ Ollama server not reachable at $OLLAMA_HOST${NC}"
    echo "    Please start Ollama server with: ollama serve"
    echo "    Or set OLLAMA_HOST environment variable"
    exit 1
fi

# Check for GuideLLM
if [ "$ENABLE_GUIDELLM" = "true" ]; then
    if command -v guidellm &> /dev/null; then
        echo -e "${GREEN}  ✓ GuideLLM is installed${NC}"
    else
        echo -e "${YELLOW}  ⚠ GuideLLM not found. Install with: pip install guidellm[recommended]${NC}"
        echo -e "${YELLOW}    Will attempt auto-install during benchmark.${NC}"
    fi
fi

echo ""

# ============================================
# Phase 2: Ollama Benchmark (Baseline)
# ============================================
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Phase 2: Ollama Benchmark${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
python3 scripts/ollama_benchmark.py \
    --host "$OLLAMA_HOST" \
    --model "$MODEL" \
    --output "$OUTPUT_DIR/ollama/results.json" \
    --num-runs "$NUM_REQUESTS" \
    --verbose

# ============================================
# Phase 3: Etalon Benchmark
# ============================================
if [ "$ENABLE_ETALON" = "true" ]; then
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Phase 3: Etalon Benchmark${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    python3 scripts/etalon_benchmark.py \
        --model "$MODEL" \
        --api-base "${OLLAMA_HOST}/v1" \
        --output-dir "$OUTPUT_DIR/etalon" \
        --max-requests "$NUM_REQUESTS" \
        --verbose || echo -e "${RED}Etalon benchmark failed (may require OpenAI-compatible API)${NC}"
fi

# ============================================
# Phase 4: GuideLLM Benchmark
# ============================================
if [ "$ENABLE_GUIDELLM" = "true" ]; then
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Phase 4: GuideLLM Benchmark${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Standard text benchmark with sweep profile
    echo -e "${YELLOW}Running GuideLLM text benchmark (sweep)...${NC}"
    python3 scripts/guidellm_benchmark.py \
        --target "${OLLAMA_HOST}/v1" \
        --model "$MODEL" \
        --profile sweep \
        --max-requests "$NUM_REQUESTS" \
        --data "prompt_tokens=256,output_tokens=128" \
        --output-dir "$OUTPUT_DIR/guidellm" \
        --verbose || echo -e "${RED}GuideLLM text benchmark failed${NC}"

    # Multimodal benchmark (text prompts designed for multimodal analysis)
    echo ""
    echo -e "${YELLOW}Running GuideLLM multimodal benchmark...${NC}"
    python3 scripts/guidellm_benchmark.py \
        --target "${OLLAMA_HOST}/v1" \
        --model "$MODEL" \
        --multimodal \
        --max-requests "$NUM_REQUESTS" \
        --output-dir "$OUTPUT_DIR/guidellm/multimodal" \
        --verbose || echo -e "${RED}GuideLLM multimodal benchmark failed${NC}"
fi

# ============================================
# Phase 5: Reasoning Benchmark
# ============================================
if [ "$ENABLE_REASONING" = "true" ]; then
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Phase 5: Reasoning Benchmark${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Run subset of reasoning tasks (configurable)
    REASONING_TASKS="${REASONING_TASKS:-3}"
    python3 scripts/reasoning_benchmark.py \
        --host "$OLLAMA_HOST" \
        --model "$MODEL" \
        --tasks "prompts/reasoning_tasks.json" \
        --max-tasks "$REASONING_TASKS" \
        --output "$OUTPUT_DIR/reasoning/results.json" \
        --verbose || echo -e "${RED}Reasoning benchmark failed${NC}"
fi

# ============================================
# Phase 6: Compare Results
# ============================================
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Phase 6: Compare Results${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
if [ -f "$OUTPUT_DIR/ollama/results.json" ]; then
    python3 scripts/compare_benchmarks.py \
        --ollama-results "$OUTPUT_DIR/ollama/results.json" \
        --etalon-results "$OUTPUT_DIR/etalon" \
        --guidellm-results "$OUTPUT_DIR/guidellm" \
        --output "$OUTPUT_DIR/comparison.json" \
        --visualize \
        --viz-output "./visualizations/comparison_$(date +%Y%m%d_%H%M%S).png" \
        || echo -e "${RED}Comparison failed${NC}"
else
    echo -e "${RED}No Ollama results to compare${NC}"
fi

# ============================================
# Phase 7: PInsight Trace Correlation (Optional)
# ============================================
if [ "$ENABLE_PINSIGHT" = "true" ]; then
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Phase 7: PInsight Trace Correlation${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    PINSIGHT_ARGS="--benchmark-results $OUTPUT_DIR/ollama/results.json"
    PINSIGHT_ARGS="$PINSIGHT_ARGS --model $MODEL"
    PINSIGHT_ARGS="$PINSIGHT_ARGS --output $OUTPUT_DIR/pinsight_correlation.json"

    if [ -n "$LTTNG_TRACE_DIR" ]; then
        PINSIGHT_ARGS="$PINSIGHT_ARGS --lttng-trace $LTTNG_TRACE_DIR"
    fi

    if [ -n "$NSIGHT_TRACE" ]; then
        PINSIGHT_ARGS="$PINSIGHT_ARGS --nsight-trace $NSIGHT_TRACE"
    fi

    python3 scripts/pinsight_correlator.py $PINSIGHT_ARGS \
        || echo -e "${RED}PInsight correlation failed${NC}"
fi

# ============================================
# Summary
# ============================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files:"
find "$OUTPUT_DIR" -type f -name "*.json" -o -name "*.csv" -o -name "*.html" 2>/dev/null | sort | while read f; do
    size=$(du -h "$f" | cut -f1)
    echo "  $size  $f"
done
echo ""

# List available visualizations
if [ -d "./visualizations" ] && [ "$(ls -A ./visualizations 2>/dev/null)" ]; then
    echo "Visualizations:"
    ls -la ./visualizations/ 2>/dev/null
fi

echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  - Review results in $OUTPUT_DIR/"
echo "  - For PInsight correlation, re-run with:"
echo "    ENABLE_PINSIGHT=true LTTNG_TRACE_DIR=/path/to/trace ./run_benchmark.sh"
echo "  - For GuideLLM HTML report, check $OUTPUT_DIR/guidellm/benchmarks.html"
echo ""
