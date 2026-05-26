#!/bin/bash
# ============================================================
# PInsight — Aries Server Setup Script
# ============================================================
# Run this on the Aries server to set up the full PInsight
# environment and run baseline vLLM experiments.
#
# Usage:
#   chmod +x setup_aries.sh
#   ./setup_aries.sh
# ============================================================

set -e

echo "============================================================"
echo "PInsight Aries Server Setup"
echo "============================================================"
echo ""

# --- Step 1: Check GPU ---
echo "[1/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "  ✓ GPU detected"
else
    echo "  ✗ nvidia-smi not found. Is CUDA installed?"
    exit 1
fi
echo ""

# --- Step 2: Create virtual environment ---
echo "[2/6] Creating Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi
source venv/bin/activate
pip install --upgrade pip -q
echo ""

# --- Step 3: Install dependencies ---
echo "[3/6] Installing dependencies..."
pip install -q requests matplotlib numpy
pip install -q vllm
echo "  ✓ Dependencies installed"
echo ""

# --- Step 4: Verify PInsight tools ---
echo "[4/6] Verifying PInsight tools..."
python3 workload_classifier.py 2>&1 | head -5
python3 tuning_engine.py --compare-all 2>&1 | head -3
echo "  ✓ PInsight tools verified"
echo ""

# --- Step 5: Create workloads if missing ---
echo "[5/6] Setting up workloads..."
python3 run_experiments.py --phase setup 2>&1 | head -6
echo "  ✓ Workloads ready"
echo ""

# --- Step 6: Run dry-run test ---
echo "[6/6] Dry-run test..."
python3 run_experiments.py --dry-run 2>&1 | tail -15
echo ""
echo "  ✓ Dry run passed"

echo ""
echo "============================================================"
echo "Setup complete! Next steps:"
echo ""
echo "  # Run baseline experiments:"
echo "  source venv/bin/activate"
echo "  python3 baseline_sweep.py --model Qwen/Qwen2.5-7B-Instruct"
echo ""
echo "  # Or run the full pipeline:"
echo "  python3 run_experiments.py --model Qwen/Qwen2.5-7B-Instruct"
echo "============================================================"
