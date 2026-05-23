#!/bin/bash
#SBATCH --job-name=hpc-bench
#SBATCH --output=logs/hpc_bench_%j.log
#SBATCH --error=logs/hpc_bench_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


set -euo pipefail

PROJECT_ROOT=$(pwd)
RESULTS_DIR="${PROJECT_ROOT}/results/hpc"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# How many GPUs were actually allocated by Slurm?
N_GPUS="${SLURM_GPUS_ON_NODE:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ "${N_GPUS}" == *,* ]]; then
    N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
fi
N_GPUS=${N_GPUS:-1}
echo "[bench] allocated GPUs: ${N_GPUS}"

# vLLM container — adjust if your AI-LAB image path differs.
VLLM_SIF="/ceph/container/vllm-openai_latest.sif"
MODEL="Qwen/Qwen2.5-7B-Instruct"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_PROFILE="${VLLM_PROFILE:-safe}"   # safe | balanced | aggressive

case "${VLLM_PROFILE}" in
    safe)
        DEFAULT_MAX_MODEL_LEN=2048
        DEFAULT_GPU_MEM_UTIL=0.75
        ;;
    balanced)
        DEFAULT_MAX_MODEL_LEN=6144
        DEFAULT_GPU_MEM_UTIL=0.80
        ;;
    aggressive)
        DEFAULT_MAX_MODEL_LEN=8192
        DEFAULT_GPU_MEM_UTIL=0.85
        ;;
    *)
        echo "[bench] invalid VLLM_PROFILE='${VLLM_PROFILE}' (expected: safe|balanced|aggressive)"
        exit 1
        ;;
esac

# Explicit env vars override profile defaults.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-${DEFAULT_GPU_MEM_UTIL}}"
BASE_PORT=8000

echo "[bench] vLLM profile=${VLLM_PROFILE} max_model_len=${MAX_MODEL_LEN} gpu_mem_util=${GPU_MEM_UTIL}"

# ---------------------------------------------------------------------------
# Launch one vLLM server per GPU
# ---------------------------------------------------------------------------

VLLM_PIDS=()
ENDPOINTS=()

for i in $(seq 0 $((N_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    LOG="${LOG_DIR}/vllm_gpu${i}.log"
    echo "[bench] starting vLLM on GPU ${i}, port ${PORT} (log: ${LOG})"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=${i} \
    singularity exec --nv "${VLLM_SIF}" \
        "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --port "${PORT}" \
        --dtype float16 \
        --enforce-eager \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        > "${LOG}" 2>&1 &
    VLLM_PIDS+=($!)
    ENDPOINTS+=("http://localhost:${PORT}/v1")
done

# Cleanup hook: kill all vLLM processes on script exit (success or failure).
cleanup() {
    echo "[bench] tearing down vLLM processes: ${VLLM_PIDS[*]}"
    for pid in "${VLLM_PIDS[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Wait for all vLLM servers to be healthy
# ---------------------------------------------------------------------------

echo "[bench] waiting for ${#ENDPOINTS[@]} vLLM endpoints to become healthy..."
for idx in "${!ENDPOINTS[@]}"; do
    ep="${ENDPOINTS[$idx]}"
    health="${ep}/models"
    deadline=$(( $(date +%s) + 600 ))   # 10-minute model-load timeout
    while true; do
        # Fail fast when the corresponding vLLM process crashed (e.g., CUDA OOM).
        pid="${VLLM_PIDS[$idx]}"
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[bench]   ${ep}  FAILED (vLLM process ${pid} exited early; check ${LOG_DIR}/vllm_gpu${idx}.log)"
            exit 3
        fi
        if curl -sf "${health}" >/dev/null 2>&1; then
            echo "[bench]   ${ep}  READY"
            break
        fi
        if [[ $(date +%s) -gt ${deadline} ]]; then
            echo "[bench]   ${ep}  TIMEOUT after 10 min — aborting"
            exit 2
        fi
        sleep 5
    done
done

# ---------------------------------------------------------------------------
# Experiment 1 — Task parallelism (intra-patient, single GPU only)
# ---------------------------------------------------------------------------

echo "[bench] === Experiment 1: task parallelism (single vLLM, varied workers) ==="

OPENAI_API_KEY=not-needed \
OPENAI_BASE_URL="${ENDPOINTS[0]}" \
LLM_PROVIDER=openai \
LLM_MODEL="${MODEL}" \
singularity exec --nv "${VLLM_SIF}" \
    "${PYTHON_BIN}" -m src.evaluation.benchmark_hpc \
        --experiment 1 \
        --n-patients 16 \
        --max-workers-list 1,2,4 \
        --output "${RESULTS_DIR}/exp1_task_parallelism.json"

echo "[bench] DONE — results in ${RESULTS_DIR}"
