# Agentic Multimodal Lung Cancer AI

## Overview
This repository contains the codebase for a Multi-Agent System (MAS) designed to predict lung cancer survival (Disease-Specific Survival) using the TCGA-LUAD and TCGA-LUSC datasets.

In clinical reality, patient records are often incomplete. This project addresses the challenge of missing multimodal data (Clinical, Transcriptomics, Whole Slide Images, and Methylation) by dynamically imputing missing modalities through an agentic workflow powered by LangGraph. The system leverages an AFM2-inspired multi-agent pipeline to detect missing data, reason about cross-modal relationships, generate plausible imputations, verify their quality, and produce survival predictions.

## System Architecture
The orchestrator is built as a Directed Acyclic Graph (DAG) using LangGraph. Data flows between nodes via a strictly typed `PatientState`. The pipeline consists of six nodes with two conditional branching points:

```
DataLoader → Planner → Miner → Generator → Verifier → Predictor
                  |                             |
                  | (all modalities present)     | (quality check failed,
                  +-----------------------------+   max 3 attempts)
                  |                             |
                  +→ Predictor                  +→ Generator (retry)
```

1. **DataLoader:** Entry point. Loads a patient's multimodal record from the preloaded TCGA cohort data and populates the shared state with raw feature arrays and modality availability flags.
2. **Planner:** Inspects which modalities are available and which are missing. Routes to the Miner if any modality is absent, or directly to the Predictor if all modalities are present.
3. **Miner:** Generates cross-modal mining rules that guide the Generator on how to reconstruct each missing modality from the available ones (AFM2-inspired). Currently uses deterministic domain rules; in production, this node will call a reasoning LLM.
4. **Generator:** Produces imputed feature arrays for each missing modality, conditioned on the Miner's rules. Currently uses zero-imputation as a mock baseline; future versions will employ modality-specific generative models.
5. **Verifier:** Assesses the quality of generated data by scoring each imputed modality. Implements a self-refinement loop: if any score falls below the threshold, execution routes back to the Generator (up to 3 attempts). Provides correction hints to guide subsequent generation cycles.
6. **Predictor:** Produces the final survival prediction using all available and imputed modalities.

## Baseline
A baseline ML pipeline (`src/baseline/`) establishes a lower-bound reference using selectable imputation strategies and survival models, followed by PCA-50. The default run uses zero-imputation + Cox Proportional Hazards (scikit-survival), and the CLI also supports KNN, KNN-tuned, MICE, CoxNet, RSF, RSF-tuned, and XGBoost. It reports the Harrell C-index overall and stratified by data completeness (complete vs. incomplete modalities), along with Kaplan-Meier survival curves.

## Repository Structure

```text
project/
├── data/                  # IGNORED BY GIT — Local storage for .pkl, .json, and .zip files
├── notebooks/             # Jupyter notebooks for EDA and prototyping
├── scripts/               # Utility scripts and batch runners
├── src/
│   ├── baseline/          # ML baseline for survival prediction comparison
│   │   ├── main_baseline.py   # Full pipeline: Load → Zero Imp → PCA-50 → CoxPH → C-index
│   │   ├── models.py          # CoxPHBaseline wrapper around scikit-survival
│   │   └── preprocessing.py   # Zero-imputation and structured array formatting
│   ├── orchestrator/      # LangGraph multi-agent system
│   │   ├── nodes/         # Individual node logic
│   │   │   ├── planner.py     # Routing decision based on modality availability
│   │   │   ├── miner.py       # Cross-modal mining rule generation (AFM2)
│   │   │   ├── generator.py   # Missing modality imputation
│   │   │   ├── verifier.py    # Quality scoring and self-refinement gating
│   │   │   ├── predictor.py   # Survival prediction output
│   │   │   └── router.py      # Conditional edge functions for DAG branching
│   │   ├── graph.py       # DAG definition, node wiring, and conditional routing
│   │   └── state.py       # TypedDict defining the shared PatientState contract
│   └── data_loader.py     # Multimodal data ingestion, shape validation, and split management
├── tests/                 # Integration and unit tests (pytest)
├── results/               # IGNORED BY GIT — Generated plots and JSON metric logs
├── requirements.txt       # Pinned project dependencies
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
Dataset files are excluded from version control. Place raw data files (`cache_data.zip` and the extracted `.pkl` / `.json` files) inside the `data/` directory:

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
python -m src.baseline.main_baseline
```
Outputs C-index metrics and diagnostic plots to `results/`.

To run the full batch of model/imputation combinations, use:
```bash
python scripts/batch_baseline.py
```

### Run the Orchestrator (Smoke Test)
```bash
python scripts/orchestrator.py
```
This script runs a small smoke test over LUAD patients and one patient with missing RNA.

## Testing
```bash
pytest tests/ -v
```