#!/bin/bash
# =============================================================================
# scripts/run_evaluation_n5.sh
# Run end-to-end orchestrator evaluation with n_candidates=5
# Usage: sbatch --gres=gpu:1 scripts/run_evaluation_n5.sh
# =============================================================================
#SBATCH --job-name=eval_n5
#SBATCH --output=logs/eval_n5_%j.log
#SBATCH --error=logs/eval_n5_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00

set -euo pipefail

CONTAINER=/ceph/container/vllm-openai_latest.sif
WORKDIR=~/agentic-multimodal-lung-cancer-ai
VLLM_PORT=8000
VLLM_LOG=logs/vllm_eval_n5_${SLURM_JOB_ID}.log
RUN_ID="n5_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
EVAL_DIR="results/evaluation/runs/n5_${SLURM_JOB_ID}"
MIN_FREE_MB=10000

cd "$WORKDIR"
mkdir -p logs "$EVAL_DIR"

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
    if [ -n "${VLLM_PID:-}" ]; then
        echo "[$(date)] Shutting down vLLM (PID=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[$(date)] GPU status:"
nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv || true

GPU_ID_RAW="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}"
GPU_ID="$(echo "$GPU_ID_RAW" | cut -d',' -f1 | sed -E 's/[^0-9]*([0-9]+).*/\1/')"
if [ -n "$GPU_ID" ]; then
    FREE_MB="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
    if [ -z "$FREE_MB" ]; then
        echo "[WARN] Could not read free memory for GPU ${GPU_ID}; skipping pre-check."
    elif [ "$FREE_MB" -lt "$MIN_FREE_MB" ]; then
        echo "[ERROR] Assigned GPU ${GPU_ID} has only ${FREE_MB} MiB free (< ${MIN_FREE_MB} MiB)."
        exit 1
    else
        echo "[$(date)] Assigned GPU ${GPU_ID} free memory: ${FREE_MB} MiB (ok)"
    fi
else
    echo "[WARN] Could not determine assigned GPU; skipping pre-check."
fi

# ── Start vLLM ───────────────────────────────────────────────────────────────
echo "[$(date)] Starting vLLM on port $VLLM_PORT"
singularity exec --nv "$CONTAINER" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.75 \
    --dtype float16 \
    --max-model-len 2048 \
    --enforce-eager \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# ── Wait for vLLM health ──────────────────────────────────────────────────────
echo "[$(date)] Waiting for vLLM to be ready..."
MAX_WAIT=1500
WAITED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "[ERROR] vLLM did not start within ${MAX_WAIT}s"
        cat "$VLLM_LOG" | tail -30
        exit 1
    fi
done
echo "[$(date)] vLLM ready after ${WAITED}s"

# ── Export LLM env vars ───────────────────────────────────────────────────────
export LLM_PROVIDER=openai
export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export OPENAI_API_KEY=not-needed
export OPENAI_BASE_URL=http://localhost:${VLLM_PORT}/v1

# ── Run evaluation ────────────────────────────────────────────────────────────
for COHORT in luad lusc; do
    echo "[$(date)] Running evaluation on ${COHORT^^}"
    singularity exec --nv "$CONTAINER" \
        python3 -m src.evaluation.evaluate_orchestrator \
        --cohort "$COHORT" \
        --model coxnet \
        --imputation mice \
        --n-candidates 5 \
        --output-dir "$EVAL_DIR" \
    || echo "[WARN] Evaluation for ${COHORT^^} failed — continuing"
    echo "[$(date)] ${COHORT^^} done"
done

echo "[$(date)] All evaluations complete"
echo ""
echo "=== RESULTS ==="
for f in "$EVAL_DIR"/cindex_comparison_*.json; do
    echo ""
    echo "--- $f ---"
    python3 -c "import json; d=json.load(open('$f')); \
        print(f'  Cohort:              {d[\"cohort\"].upper()}'); \
        print(f'  N test:              {d[\"n_test\"]}'); \
        print(f'  N missing:           {d.get(\"n_missing\", \"N/A\") }'); \
        print(f'  Baseline C-index:    {d.get(\"baseline_cindex\", \"N/A\")}'); \
        print(f'  Orchestrator C-idx:  {d.get(\"orchestrator_cindex\", \"N/A\")}'); \
        print(f'  Delta:               {d.get(\"delta_cindex\", \"N/A\")}'); \
        print(f'  Mean provenance:     {d.get(\"mean_provenance\", \"N/A\")}'); \
        print(f'  Baseline source:     {d.get(\"baseline_source\", \"N/A\")}')"
done
