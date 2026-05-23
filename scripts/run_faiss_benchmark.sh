#!/bin/bash
#SBATCH --job-name=faiss_bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/ceph/home/student.aau.dk/sq02qe/agentic-multimodal-lung-cancer-ai
#SBATCH --output=/ceph/home/student.aau.dk/sq02qe/agentic-multimodal-lung-cancer-ai/faiss_bench_%j.log
#SBATCH --error=/ceph/home/student.aau.dk/sq02qe/agentic-multimodal-lung-cancer-ai/faiss_bench_err_%j.log

set -euo pipefail

export MAMBA_ROOT_PREFIX="${HOME}/micromamba-root"
export PYTHONNOUSERSITE=1

# Canonical parameters matching generator.py defaults (BASE_K=5, DEFAULT_N_CANDIDATES=3)
K=20
N_CANDIDATES=3
N_QUERIES=100

MICROMAMBA="${HOME}/.local/bin/micromamba"
ENV="faissgpu"

echo "[$(date)] GPU status:"
nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv || true

for COHORT in luad lusc; do
    OUT="results/benchmarks/faiss_comparison_${COHORT}_k${K}_n${N_CANDIDATES}.json"
    echo "[$(date)] Benchmarking ${COHORT^^} (k=${K}, n_candidates=${N_CANDIDATES}, n_queries=${N_QUERIES})..."
    "$MICROMAMBA" run -n "$ENV" python -m src.evaluation.benchmark_faiss \
        --cohort "$COHORT" \
        --k "$K" \
        --n-candidates "$N_CANDIDATES" \
        --n-queries "$N_QUERIES"
    echo "[$(date)] ${COHORT^^} done -> ${OUT}"
done

echo ""
echo "=== Results ==="
ls -lh results/benchmarks/faiss_comparison_*_k${K}_n${N_CANDIDATES}.json
