#!/bin/bash
#SBATCH --job-name=hpc-analyze
#SBATCH --output=logs/analyze_hpc_%j.log
#SBATCH --error=logs/analyze_hpc_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# ---------------------------------------------------------------------------
# analyze_hpc.sh — CPU-only job to generate report and plots for Exp1
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT=$(pwd)
RESULTS_DIR="${PROJECT_ROOT}/results/hpc"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[analyze] running HPC analysis..."
"${PYTHON_BIN}" -m src.evaluation.analyze_hpc \
    --inputs "${RESULTS_DIR}/exp1_task_parallelism.json" \
    --plot-dir "${RESULTS_DIR}/plots" \
    --report "${RESULTS_DIR}/hpc_summary.md"

echo "[analyze] DONE — report and plots in ${RESULTS_DIR}"
