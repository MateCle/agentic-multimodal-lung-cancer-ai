#!/bin/bash
# =============================================================================
# scripts/run_evaluation_t0.sh  (v2 — conservative startup)
#
# T=0 GENERATOR ABLATION RUN
# --------------------------
# Changes vs v1:
#   • Removed --enforce-eager (caused engine startup crash on AI-LAB)
#   • Binds HuggingFace cache so model is reused if already downloaded
#   • Increased startup timeout to 1200s (some nodes are slow on first run)
#   • Logs more aggressively on failure so you don't have to guess
# =============================================================================

#SBATCH --job-name=eval_t0
#SBATCH --output=logs/eval_t0_%j.log
#SBATCH --error=logs/eval_t0_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

CONTAINER=/ceph/container/vllm-openai_latest.sif
WORKDIR=~/agentic-multimodal-lung-cancer-ai
VLLM_PORT=8000
VLLM_LOG=logs/vllm_eval_t0_${SLURM_JOB_ID}.log
RUN_ID="t0_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
EVAL_DIR="results/evaluation/runs/${RUN_ID}"
HF_CACHE_DIR="$HOME/.cache/huggingface"
MIN_FREE_MB=10000

cd "$WORKDIR"
mkdir -p logs "$EVAL_DIR" "$HF_CACHE_DIR"

cleanup() {
    echo "[$(date)] Cleanup — shutting down vLLM"
    if [ -n "${VLLM_PID:-}" ]; then
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

echo "[$(date)] Starting vLLM on port $VLLM_PORT"
echo "[$(date)] vLLM log: $VLLM_LOG"

singularity exec --nv \
    -B "$HF_CACHE_DIR":/root/.cache/huggingface \
    "$CONTAINER" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.75 \
    --dtype float16 \
    --max-model-len 2048 \
    --enforce-eager \
    --disable-log-requests \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "[$(date)] Waiting for vLLM to be ready (timeout: 1200s)..."
MAX_WAIT=1800
WAITED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 10
    WAITED=$((WAITED + 10))
    # Early-exit if engine has crashed
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[ERROR] vLLM process died at ${WAITED}s. Last 80 lines of log:"
        tail -80 "$VLLM_LOG"
        exit 1
    fi
    if [ $((WAITED % 60)) -eq 0 ]; then
        echo "[$(date)] still waiting... ${WAITED}s elapsed"
    fi
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "[ERROR] vLLM did not become healthy within ${MAX_WAIT}s. Log dump:"
        tail -80 "$VLLM_LOG"
        exit 1
    fi
done
echo "[$(date)] vLLM ready after ${WAITED}s"

export LLM_PROVIDER=openai
export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export OPENAI_API_KEY=not-needed
export OPENAI_BASE_URL=http://localhost:${VLLM_PORT}/v1

for COHORT in luad lusc; do
    echo ""
    echo "[$(date)] T=0 evaluation: ${COHORT^^}"
    singularity exec --nv \
        -B "$HF_CACHE_DIR":/root/.cache/huggingface \
        "$CONTAINER" \
        python3 -m src.evaluation.evaluate_orchestrator \
        --cohort "$COHORT" \
        --model coxnet \
        --imputation mice \
        --generator-temperature 0.0 \
        --output-dir "$EVAL_DIR" \
    || echo "[WARN] ${COHORT^^} evaluation failed — continuing"
done

echo ""
echo "[$(date)] Done. Results in $EVAL_DIR"