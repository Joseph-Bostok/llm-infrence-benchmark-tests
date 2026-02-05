#!/bin/bash
# LLM Inference Benchmark Runner
# Runs Ollama and Etalon benchmarks and compares results

set -e

# Configuration
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODEL="${MODEL:-llama2}"
OUTPUT_DIR="./results/$(date +%Y%m%d_%H%M%S)"
NUM_REQUESTS="${NUM_REQUESTS:-5}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}LLM Inference Benchmark Suite${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Configuration:"
echo "  Ollama Host: $OLLAMA_HOST"
echo "  Model: $MODEL"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Requests: $NUM_REQUESTS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR/ollama"
mkdir -p "$OUTPUT_DIR/etalon"
mkdir -p "./visualizations"

# Check if Ollama server is running
echo -e "${YELLOW}Checking Ollama server...${NC}"
if curl -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo -e "${GREEN}Ollama server is running${NC}"
else
    echo -e "${RED}Ollama server not reachable at $OLLAMA_HOST${NC}"
    echo "Please start Ollama server with: ollama serve"
    echo "Or set OLLAMA_HOST environment variable"
    exit 1
fi

# Run Ollama benchmark
echo ""
echo -e "${YELLOW}Running Ollama Benchmark...${NC}"
python3 scripts/ollama_benchmark.py \
    --host "$OLLAMA_HOST" \
    --model "$MODEL" \
    --output "$OUTPUT_DIR/ollama/results.json" \
    --num-runs "$NUM_REQUESTS" \
    --verbose

# Run Etalon benchmark (if server supports OpenAI API)
echo ""
echo -e "${YELLOW}Running Etalon Benchmark...${NC}"
python3 scripts/etalon_benchmark.py \
    --model "$MODEL" \
    --api-base "${OLLAMA_HOST}/v1" \
    --output-dir "$OUTPUT_DIR/etalon" \
    --max-requests "$NUM_REQUESTS" \
    --verbose || echo -e "${RED}Etalon benchmark failed (may require OpenAI-compatible API)${NC}"

# Compare results
echo ""
echo -e "${YELLOW}Comparing Results...${NC}"
if [ -f "$OUTPUT_DIR/ollama/results.json" ]; then
    python3 scripts/compare_benchmarks.py \
        --ollama-results "$OUTPUT_DIR/ollama/results.json" \
        --etalon-results "$OUTPUT_DIR/etalon" \
        --output "$OUTPUT_DIR/comparison.json" \
        --visualize \
        --viz-output "./visualizations/comparison_$(date +%Y%m%d_%H%M%S).png"
else
    echo -e "${RED}No Ollama results to compare${NC}"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files:"
ls -la "$OUTPUT_DIR/" 2>/dev/null || true
