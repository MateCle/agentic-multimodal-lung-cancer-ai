#!/bin/bash
#SBATCH --job-name=orchestrator_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=orchestrator_%j.log
#SBATCH --error=orchestrator_err_%j.log

set -euo pipefail

# Helps reduce CUDA allocator fragmentation during vLLM model load.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

VLLM_CONTAINER="/ceph/container/vllm-openai_latest.sif"
MODEL="Qwen/Qwen2.5-7B-Instruct"
MIN_FREE_MB=10000
PATIENT="${1:-TCGA-99-8033}"
TMP_OUT="/tmp/orch_out_${SLURM_JOB_ID:-$$}.txt"

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

echo "Starting vLLM server in background..."
singularity exec --nv --bind /etc/passwd:/etc/passwd --bind /etc/group:/etc/group "$VLLM_CONTAINER" \
  vllm serve "$MODEL" \
  --max-model-len 4096 \
  --dtype half \
  --port 8000 &

VLLM_PID=$!

echo "Waiting for vLLM server to start..."
for _ in $(seq 1 120); do
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "vLLM server is ready!"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM process exited before becoming ready."
    exit 1
  fi
  sleep 3
done

if ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
  echo "Timed out waiting for vLLM server startup."
  exit 1
fi

# Setup environment variables for local routing
export LLM_PROVIDER="openai"
export LLM_MODEL="$MODEL"
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"

echo "Running orchestrator for patient ${PATIENT}..."
singularity exec --nv --bind /etc/passwd:/etc/passwd --bind /etc/group:/etc/group "$VLLM_CONTAINER" \
  python3 -m src.orchestrator.run --patient "$PATIENT" --verbose \
  | tee "$TMP_OUT"

# Extract cohort from output (printed as "  Cohort             : LUAD")
COHORT=$(grep -E 'Cohort\s+:' "$TMP_OUT" | awk '{print tolower($NF)}' | head -1)
COHORT="${COHORT:-unknown}"

REPORT_MD="report_${PATIENT}_${COHORT}_${SLURM_JOB_ID:-local}.md"

{
  echo "# Clinical Report — ${PATIENT} (${COHORT^^})"
  echo "_Job: ${SLURM_JOB_ID:-local} — $(date)_"
  echo ""
  awk '/^  CLINICAL REPORT$/{f=1;next} f&&/^============================================================$/{f=0;p=1;next} p{print}' "$TMP_OUT"
} > "$REPORT_MD"

rm -f "$TMP_OUT"
echo "[$(date)] Report saved to: ${REPORT_MD}"

echo "Stopping vLLM server..."
kill "$VLLM_PID"