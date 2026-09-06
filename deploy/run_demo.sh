#!/usr/bin/env bash
# ==============================================================================
# ECG Arrhythmia Edge AI - Raspberry Pi One-Click Verification & Demo Script
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "================================================================================"
echo "         ECG ARRHYTHMIA EDGE AI - RASPBERRY PI RUNTIME SUITE"
echo "================================================================================"

# Locate and activate virtual environment
if [ -d "../.venv" ]; then
    echo "Activating virtual environment: ../.venv"
    source ../.venv/bin/activate
elif [ -d "./.venv" ]; then
    echo "Activating virtual environment: ./.venv"
    source ./.venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using currently active virtual environment: $VIRTUAL_ENV"
else
    echo "WARNING: No virtual environment detected. Using system python3."
fi

# Step 1: Run Verification Self-Test
echo ""
echo "[STEP 1/3] Running Deployment Verification & Graph Inspection..."
python3 verify_deployment.py

# Step 2: Run Synthetic Cascade Demonstration
echo ""
echo "[STEP 2/3] Running Hierarchical 3-Tier Clinical Triage Cascade..."
python3 edge_runtime.py

# Step 3: Run Edge Throughput & Latency Benchmark
echo ""
echo "[STEP 3/3] Running Edge CPU Hardware Benchmark (100 iterations)..."
python3 edge_runtime.py --benchmark

echo ""
echo "================================================================================"
echo "  ALL PI DEPLOYMENT STEPS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
