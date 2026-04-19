#!/bin/bash
#SBATCH --job-name=orchestrator_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=orchestrator_%j.log
#SBATCH --error=orchestrator_err_%j.log

VLLM_CONTAINER="/ceph/container/vllm-openai_latest.sif"
MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "Starting vLLM server in background..."
singularity exec --nv $VLLM_CONTAINER \
  vllm serve $MODEL \
  --max-model-len 4096 \
  --dtype half \
  --port 8000 &

VLLM_PID=$!

echo "Waiting for vLLM server to start..."
while ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; do
  sleep 3
done
echo "vLLM server is ready!"

# Setup environment variables for local routing
export LLM_PROVIDER="openai"
export LLM_MODEL=$MODEL
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"

echo "Running LangGraph smoke test..."
singularity exec --nv $VLLM_CONTAINER \
  python3 scripts/orchestrator.py

echo "Stopping vLLM server..."
kill $VLLM_PID