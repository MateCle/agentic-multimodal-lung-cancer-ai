#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --output=logs/exp2_%j.log
#SBATCH --error=logs/exp2_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail
CONTAINER=/ceph/container/vllm-openai_latest.sif
WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BASE_PORT=8000
MIN_FREE_MB=10000
mkdir -p logs results/hpc
cd "$WORKDIR"

echo "[$(date)] GPU status:"
nvidia-smi --query-gpu=index,name,memory.free,memory.used --format=csv || true

GPU_LIST="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}}"
GPU_IDS=()
if [ -n "$GPU_LIST" ]; then
    IFS=',' read -r -a GPU_RAW <<< "$GPU_LIST"
    for raw in "${GPU_RAW[@]}"; do
        gpu_id="$(echo "$raw" | sed -E 's/[^0-9]*([0-9]+).*/\1/')"
        if [ -n "$gpu_id" ]; then
            GPU_IDS+=("$gpu_id")
        fi
    done
else
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        for i in $(seq 0 $((GPU_COUNT - 1))); do
            GPU_IDS+=("$i")
        done
    fi
fi

N_GPUS=${#GPU_IDS[@]}
if [ "$N_GPUS" -le 0 ]; then
    N_GPUS=4
    GPU_IDS=(0 1 2 3)
fi

for gpu in "${GPU_IDS[@]}"; do
    GPU_ID="$(echo "$gpu" | sed -E 's/[^0-9]*([0-9]+).*/\1/')"
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
    fi
done

if [ "$N_GPUS" -ge 4 ]; then
    WORKERS_LIST="1,2,4"
elif [ "$N_GPUS" -ge 2 ]; then
    WORKERS_LIST="1,2"
else
    WORKERS_LIST="1"
fi

echo "SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Using GPU IDs: ${GPU_IDS[*]}"
echo "Detected N_GPUS=$N_GPUS (workers-list=$WORKERS_LIST)"

PIDS=()
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for idx in $(seq 0 $((N_GPUS-1))); do
    GPU_ID="${GPU_IDS[$idx]}"
    PORT=$((BASE_PORT + idx))
    CUDA_VISIBLE_DEVICES=$GPU_ID singularity exec --nv --bind /etc/passwd:/etc/passwd --bind /etc/group:/etc/group "$CONTAINER" \
        python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --port $PORT \
        --gpu-memory-utilization 0.75 \
        --dtype float16 \
        --max-model-len 2048 \
        --enforce-eager \
        > logs/vllm_gpu${GPU_ID}_${SLURM_JOB_ID}.log 2>&1 &
    PIDS+=($!)
    sleep 10  # Stagger launches to avoid simultaneous GPU memory allocation
done

cleanup() {
    if [ ${#PIDS[@]} -gt 0 ]; then
        kill "${PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
echo "Waiting for vLLM servers to be ready..."
for idx in $(seq 0 $((N_GPUS-1))); do
    GPU_ID="${GPU_IDS[$idx]}"
    PORT=$((BASE_PORT + idx))
    WAITED=0
    until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
        sleep 5; WAITED=$((WAITED+5))
        [ $WAITED -ge 900 ] && echo "GPU ${GPU_ID} timeout" && exit 1
    done
    echo "GPU ${GPU_ID} ready"
done

VLLM_ENDPOINTS=$(for idx in $(seq 0 $((N_GPUS-1))); do echo -n "http://localhost:$((BASE_PORT + idx))/v1,"; done | sed 's/,$//')
echo "VLLM_ENDPOINTS=$VLLM_ENDPOINTS"

singularity exec --nv --bind /etc/passwd:/etc/passwd --bind /etc/group:/etc/group "$CONTAINER" \
    python3 -m src.evaluation.benchmark_hpc \
    --experiment 2 \
    --n-patients 32 \
    --vllm-endpoints "$VLLM_ENDPOINTS" \
    --workers-list "$WORKERS_LIST" \
    --scaling strong \
    --output results/hpc/exp2_data_parallelism_${N_GPUS}gpu.json
