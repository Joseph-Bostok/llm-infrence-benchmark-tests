# LLM Inference Benchmark

A benchmarking suite for comparing LLM inference performance between Ollama and Etalon frameworks.

## Overview

This project provides tools to measure and compare LLM inference metrics:

- **Time to First Token (TTFT)**: Latency from request to first token
- **Time Per Output Token (TPOT)**: Average time per generated token
- **Inter-Token Latency (ITL)**: Time between consecutive tokens
- **Tokens Per Second**: Throughput measurement
- **Fluidity Index**: Etalon's novel metric for streaming smoothness

## Project Structure

```
llm-inference-benchmark/
├── scripts/
│   ├── ollama_benchmark.py    # Custom Ollama benchmarking with detailed timing
│   ├── etalon_benchmark.py    # Etalon framework wrapper
│   └── compare_benchmarks.py  # Comparison and visualization
├── etalon/                    # Etalon framework (cloned)
├── ollama/                    # Ollama binary (if installed locally)
├── results/                   # Benchmark results
├── visualizations/            # Generated charts
├── run_benchmark.sh           # Main runner script
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- Ollama server (local or remote)
- pip packages: ollama, matplotlib (optional)

### Setup

```bash
# Install Python dependencies
pip install ollama matplotlib

# Etalon is already installed in ./etalon/
cd etalon && pip install -e . && cd ..
```

## Usage

### Quick Start

```bash
# Set Ollama host (if not localhost)
export OLLAMA_HOST=http://localhost:11434

# Set model to benchmark
export MODEL=llama2

# Run full benchmark suite
./run_benchmark.sh
```

### Individual Benchmarks

#### Ollama Benchmark

```bash
python3 scripts/ollama_benchmark.py \
    --host http://localhost:11434 \
    --model llama2 \
    --output results/ollama_results.json \
    --verbose

# With custom prompt
python3 scripts/ollama_benchmark.py \
    --host http://localhost:11434 \
    --model llama2 \
    --prompt "Explain quantum computing in simple terms" \
    --num-runs 5 \
    --output results/ollama_results.json
```

#### Etalon Benchmark

```bash
python3 scripts/etalon_benchmark.py \
    --model llama2 \
    --api-base http://localhost:11434/v1 \
    --output-dir results/etalon \
    --max-requests 50 \
    --verbose
```

#### Compare Results

```bash
python3 scripts/compare_benchmarks.py \
    --ollama-results results/ollama_results.json \
    --etalon-results results/etalon \
    --output results/comparison.json \
    --visualize
```

## Metrics Explained

### Ollama Benchmark Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time from request start to first token received |
| TPOT | Average time per output token (total_time / total_tokens) |
| ITL Mean | Average inter-token latency (excluding first token) |
| ITL Std | Standard deviation of inter-token latencies |
| ITL P50/P90/P99 | Percentile latencies |
| Tokens/s | Output tokens per second |

### Etalon Metrics

| Metric | Description |
|--------|-------------|
| TTFT | Time to first token |
| TBT | Time between tokens (similar to ITL) |
| Fluidity Index | Novel metric measuring streaming smoothness |
| TTFT Deadline | Percentage of requests meeting TTFT deadline |
| TBT Deadline | Percentage of tokens meeting TBT deadline |

## Example Output

```
============================================================
OLLAMA INFERENCE METRICS
============================================================
Model: llama2
Prompt length: 58 chars
Output tokens: 127
------------------------------------------------------------
TIMING METRICS:
  Time to First Token (TTFT):        234.56 ms
  Total Generation Time:            2341.23 ms
  Time Per Output Token (TPOT):       18.43 ms
  Tokens Per Second:                  54.23 tok/s
------------------------------------------------------------
INTER-TOKEN LATENCY (ITL):
  Mean:                               18.12 ms
  Std Dev:                             3.45 ms
  Min:                                12.34 ms
  Max:                                34.56 ms
  P50 (Median):                       17.89 ms
  P90:                                23.45 ms
  P99:                                31.23 ms
============================================================
```

## Configuration Options

### Ollama Benchmark

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | localhost:11434 | Ollama server URL |
| `--model` | llama2 | Model to benchmark |
| `--prompt` | (default set) | Single prompt to test |
| `--prompts-file` | - | JSON file with prompts |
| `--output` | - | Output JSON file |
| `--num-runs` | 1 | Repetitions per prompt |
| `--verbose` | false | Per-token timing output |

### Etalon Benchmark

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | llama2 | Model name |
| `--api-base` | localhost:11434/v1 | OpenAI-compatible API |
| `--max-requests` | 50 | Requests to complete |
| `--num-clients` | 1 | Parallel clients |
| `--qps` | 0.5 | Queries per second |
| `--ttft-deadline` | 0.5 | TTFT deadline (seconds) |
| `--tbt-deadline` | 0.05 | TBT deadline (seconds) |

## Integration with Performance Analysis

This benchmark suite integrates with the broader in-situ performance analysis framework:

1. **Python Profiling**: Use with cProfile extensions for detailed call-tree analysis
2. **LTTng Tracing**: Correlate with system-level traces
3. **Eclipse Trace Compass**: Visualize cross-layer performance

## References

- [Ollama](https://ollama.com/) - Local LLM inference server
- [Etalon](https://github.com/project-etalon/etalon) - LLM performance evaluation framework
- [Etalon Paper](https://arxiv.org/abs/2407.07000) - "Holistic Performance Evaluation Framework for LLM Inference Systems"

## License

MIT License
# llm-infrence-benchmark-tests
