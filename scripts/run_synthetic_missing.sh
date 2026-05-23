#!/bin/bash
# =============================================================================
# scripts/run_synthetic_missing.sh
#
# Synthetic missing-modality reconstruction experiment.
# For each cohort: take test patients with ALL 4 modalities present, mask
# one modality at a time, reconstruct via orchestrator (Generator T=0),
# measure cosine sim + MSE vs ground truth, correlate with Verifier scores.
# =============================================================================

#SBATCH --job-name=syn_miss
#SBATCH --output=logs/syn_miss_%j.log
#SBATCH --error=logs/syn_miss_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

CONTAINER=/ceph/container/vllm-openai_latest.sif
WORKDIR=~/agentic-multimodal-lung-cancer-ai
VLLM_PORT=8000
VLLM_LOG=logs/vllm_syn_miss_${SLURM_JOB_ID}.log
RUN_ID="syn_miss_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/evaluation/synthetic_missing/${RUN_ID}"
HF_CACHE_DIR="$HOME/.cache/huggingface"
MIN_FREE_MB=10000
COHORTS="${COHORTS:-luad lusc}"
MAX_PATIENTS="${MAX_PATIENTS:-}"
SEED="${SEED:-42}"

cd "$WORKDIR"
mkdir -p logs "$OUTPUT_DIR" "$HF_CACHE_DIR"

cleanup() {
    echo "[$(date)] Cleanup"
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
singularity exec --nv \
    -B "$HF_CACHE_DIR":/root/.cache/huggingface \
    "$CONTAINER" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.90 \
    --dtype float16 \
    --max-model-len 2048 \
    --enforce-eager \
    --disable-log-requests \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "[$(date)] Waiting for vLLM (timeout 1200s)..."
MAX_WAIT=1800
WAITED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 10
    WAITED=$((WAITED + 10))
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[ERROR] vLLM died at ${WAITED}s"
        tail -80 "$VLLM_LOG"
        exit 1
    fi
    [ $((WAITED % 60)) -eq 0 ] && echo "[$(date)] waiting ${WAITED}s..."
    [ "$WAITED" -ge "$MAX_WAIT" ] && { echo "[ERROR] timeout"; tail -80 "$VLLM_LOG"; exit 1; }
done
echo "[$(date)] vLLM ready after ${WAITED}s"

export LLM_PROVIDER=openai
export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export OPENAI_API_KEY=not-needed
export OPENAI_BASE_URL=http://localhost:${VLLM_PORT}/v1

if [ -n "$MAX_PATIENTS" ]; then
    echo "[$(date)] Limiting to MAX_PATIENTS=$MAX_PATIENTS (SEED=$SEED)"
fi

for COHORT in $COHORTS; do
    echo ""
    echo "[$(date)] Synthetic missing reconstruction: ${COHORT^^}"
    singularity exec --nv \
        -B "$HF_CACHE_DIR":/root/.cache/huggingface \
        "$CONTAINER" \
        python3 -m src.evaluation.synthetic_missing_eval \
        --cohort "$COHORT" \
        ${MAX_PATIENTS:+--max-patients "$MAX_PATIENTS"} \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" \
    || echo "[WARN] ${COHORT^^} failed — continuing"
done

echo ""
echo "[$(date)] Done. Results: $OUTPUT_DIR"
echo ""
echo "=== Summary ==="
for f in "$OUTPUT_DIR"/summary_*.json; do
    [ -f "$f" ] || continue
    echo "--- $f ---"
    python3 -c "
import json
d = json.load(open('$f'))
print(f'  Cohort: {d[\"cohort\"].upper()}  N patients: {d[\"n_patients_evaluated\"]}  N recons: {d[\"n_reconstructions\"]}')
for m, stats in d['reconstruction']['per_modality'].items():
    n = stats.get('n_samples', 0)
    if n == 0:
        print(f'    {m:<16} N=0')
        continue
    cos = stats['cosine_similarity']['mean']
    nmse = stats['normalized_mse']['mean']
    print(f'    {m:<16} N={n}  cos={cos:.3f}  norm.MSE={nmse:.3f}')
v = d['verifier_validation']
print(f'  Verifier corr (Pearson cos):    {v.get(\"pearson_r_cos\")}')
print(f'  Verifier corr (Spearman cos):   {v.get(\"spearman_r_cos\")}')
"
done