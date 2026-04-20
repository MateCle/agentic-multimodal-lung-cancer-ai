# Agentic Multimodal Lung Cancer AI

## Overview
This repository contains the codebase for a Multi-Agent System (MAS) designed to predict lung cancer survival (Disease-Specific Survival) using the TCGA-LUAD and TCGA-LUSC datasets.

In clinical reality, patient records are often incomplete. This project addresses the challenge of missing multimodal data (Clinical, Transcriptomics, Whole Slide Images, and Methylation) by dynamically imputing missing modalities through an agentic workflow powered by LangGraph. The system leverages an AFM2-inspired multi-agent pipeline to detect missing data, reason about cross-modal relationships, generate plausible imputations, verify their quality, and produce survival predictions.

## System Architecture
The orchestrator is built as a LangGraph stateful directed graph with bounded self-refinement. Data flows between nodes via a strictly typed `PatientState`. The pipeline consists of six nodes with two conditional branching points:

```
DataLoader → Planner → Miner → Generator → Verifier → Predictor
                  |                             |
                  | (all modalities present)     | (quality check failed,
                  +→ Predictor                  +→ Generator (retry,
                                                   max 3 attempts)
```

1. **DataLoader:** Entry point. Loads a patient's multimodal record from the preloaded TCGA cohort data and populates the shared state with raw feature arrays and modality availability flags.
2. **Planner:** Inspects which modalities are available and which are missing. Routes to the Miner if any modality is absent, or directly to the Predictor if all modalities are present.
3. **Miner (LLM):** Calls an LLM (Qwen2.5-7B-Instruct via vLLM) to reason about cross-modal biological relationships and produce mining rules for each missing modality, following AFM2's Miner Agent pattern.
4. **Generator (LLM + k-NN):** Uses the LLM to interpret mining rules into modality weights, then performs k-NN retrieval over the training pool with weighted cosine similarity to reconstruct missing features.
5. **Verifier (LLM):** Performs a distributional check followed by LLM-based multi-criteria quality scoring (6 clinical criteria, each 0-5, following AFM2). Implements a self-refinement loop: if the overall score falls below threshold (4.0), execution routes back to the Generator with correction hints (up to 3 attempts).
6. **Predictor:** Assembles all features (real + generated) and runs the fitted baseline pipeline (scaler → PCA → survival model) for the final risk score prediction.

### LLM Usage
Three nodes use the LLM (Miner, Generator, Verifier). The remaining nodes (DataLoader, Planner, Predictor) are deterministic. The system supports three LLM providers via a unified client:
- **Local vLLM** (Qwen2.5-7B-Instruct on AAU AI-LAB) — primary mode
- **OpenAI API** (GPT-4o) — alternative
- **Mock** — deterministic responses for testing without GPU

## Baseline
A baseline ML pipeline (`src/baseline/`) establishes a reference performance using selectable imputation strategies and survival models, followed by PCA-50 dimensionality reduction. The CLI supports:
- **Imputation:** zero, KNN, KNN-tuned, MICE
- **Models:** CoxPH, CoxNet, RSF, XGBoost

Evaluation uses the Harrell C-index as the sole metric (binary AUC is methodologically invalid for right-censored survival data). Results are reported overall and stratified by modality completeness (complete vs. incomplete patients), along with Kaplan-Meier survival curves and SHAP explainability with PCA back-projection.

## Repository Structure

```text
project/
├── data/                      # IGNORED BY GIT — .pkl and .json data files
├── models/                    # IGNORED BY GIT — Fitted pipeline .joblib files
├── notebooks/                 # Jupyter notebooks for EDA and prototyping
├── scripts/                   # Utility and SLURM scripts
│   └── run_orchestrator.sh    # SLURM batch script for AI-LAB (vLLM + orchestrator)
├── src/
│   ├── data_loader.py         # Multimodal data ingestion, shape validation, splits
│   ├── baseline/              # ML baseline for survival prediction
│   │   ├── main_baseline.py   # Full pipeline: Load → Impute → PCA-50 → Model → C-index
│   │   ├── models.py          # CoxPH, CoxNet, RSF, XGBoost survival models
│   │   ├── preprocessing.py   # Imputation strategies (zero, KNN, MICE) + feature matrix
│   │   ├── explain.py         # SHAP with PCA back-projection to original features
│   │   └── pipeline.py        # Serialize/load fitted (model, scaler, PCA) pipelines
│   └── orchestrator/          # LangGraph multi-agent system
│       ├── llm.py             # Unified LLM client (OpenAI, Anthropic, vLLM, Mock)
│       ├── graph.py           # DAG definition, node wiring, conditional routing
│       ├── state.py           # TypedDict defining the shared PatientState
│       ├── run.py             # CLI entry point for the orchestrator
│       └── nodes/             # Individual node implementations
│           ├── planner.py     # Routing decision based on modality availability
│           ├── miner.py       # LLM-based cross-modal mining rule generation
│           ├── generator.py   # LLM-guided k-NN retrieval for missing modalities
│           ├── verifier.py    # Multi-criteria LLM scoring + self-refinement
│           ├── predictor.py   # Survival prediction via fitted baseline pipeline
│           └── router.py      # Conditional edge functions for DAG branching
├── tests/                     # Unit and integration tests (pytest)
├── results/                   # JSON metric logs (committed), PNG plots (gitignored)
├── requirements.txt           # Project dependencies
└── README.md
```

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/MateCle/agentic-multimodal-lung-cancer-ai.git
cd agentic-multimodal-lung-cancer-ai
```

### 2. Python Environment Setup
Python 3.10+ is required. Use either `venv` or `conda`:

**Option A: venv**
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

**Option B: Conda**
```bash
conda create -n amlc python=3.10
conda activate amlc
pip install -r requirements.txt
```

### 3. Data Placement
Dataset files are excluded from version control. Place raw data files inside the `data/` directory:

```text
data/
└── extracted/
    └── cache_data/
        ├── tcga_luad_prepared_data.pkl
        ├── tcga_luad_metadata.pkl
        ├── tcga_lusc_prepared_data.pkl
        ├── tcga_lusc_metadata.pkl
        └── splits/
            ├── tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json
            └── tcga_lusc_DSS_k5_r1_test0.2_val0.2_seed42.json
```

## Usage

### Run the ML Baseline
```bash
# Default: CoxPH + zero imputation
python -m src.baseline.main_baseline

# Specific model + imputation + SHAP
python -m src.baseline.main_baseline --model xgboost --imputation knn --shap
```
Outputs C-index metrics to `results/` and diagnostic plots.

### Run the Orchestrator

**Mock mode (no GPU, no LLM):**
```bash
python -m src.orchestrator.run --patient TCGA-05-4244 --verbose --mock
```

**Real mode (requires vLLM server running):**
```bash
python -m src.orchestrator.run --patient TCGA-05-4244 --verbose
```

**Multiple patients:**
```bash
python -m src.orchestrator.run --n-patients 5 --verbose
```

## AI-LAB Setup (AAU HPC)

The orchestrator uses Qwen2.5-7B-Instruct served via vLLM on AAU's AI-LAB infrastructure (Slurm + Singularity).

### Prerequisites
- AAU student account with AI-LAB access
- Hugging Face access token ([create one here](https://huggingface.co/settings/tokens))

### One-time setup
```bash
# SSH into AI-LAB
ssh <your-id>@ai-lab.aau.dk

# Configure HF token
echo 'export HF_TOKEN="YOUR_TOKEN_HERE"' >> ~/.bashrc
source ~/.bashrc

# Clone the repo
git clone https://github.com/MateCle/agentic-multimodal-lung-cancer-ai.git
cd agentic-multimodal-lung-cancer-ai

# Upload data from your local machine (run this locally, not on AI-LAB)
scp -r data/extracted/cache_data <your-id>@ai-lab.aau.dk:~/agentic-multimodal-lung-cancer-ai/data/extracted/

# Install Python dependencies (on AI-LAB, inside a GPU job)
srun --gres=gpu:1 --mem=24G --time=00:10:00 \
  singularity exec --nv /ceph/container/vllm-openai_latest.sif \
  pip install --user -r requirements.txt
```

### Running the orchestrator on AI-LAB
```bash
sbatch scripts/run_orchestrator.sh
tail -f orchestrator_*.log
```

The SLURM script (`scripts/run_orchestrator.sh`) automatically:
1. Starts a vLLM server with Qwen2.5-7B-Instruct on a GPU
2. Waits for the server to be ready
3. Runs the orchestrator with LLM-based reasoning
4. Stops the server after completion

### Resource limits
- Max 24 GB GPU memory per GPU, 15 CPUs per GPU, 8 GPUs per user
- Qwen2.5-7B-Instruct requires ~14 GB VRAM in half precision (fits on 1 GPU)
- For CUDA OOM errors, reduce `--max-model-len` to 2048 in the SLURM script

## Testing
```bash
pytest tests/ -v
```

## Data Classification
TCGA data is Level 1 (publicly available, de-identified) under AAU's data classification model. AI-LAB is appropriate for this data.