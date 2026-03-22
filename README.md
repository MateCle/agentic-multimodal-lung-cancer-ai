# Agentic Multimodal Lung Cancer AI

## Overview
This repository contains the codebase for a Multi-Agent System (MAS) designed to predict lung cancer survival (Disease-Specific Survival and Overall Survival) using the TCGA-LUAD and TCGA-LUSC datasets. 

In clinical reality, patient records are often incomplete. This project addresses the challenge of missing multimodal data (Clinical, Transcriptomics, Whole Slide Images, and Methylation) by dynamically imputing missing modalities through an agentic workflow powered by LangGraph. The system leverages a multi-agent approach to handle clinical tensor data, evaluate biological plausibility, and dynamically adapt to missing information through a self-refinement loop.

## System Architecture
The orchestrator is built as a Directed Acyclic Graph (DAG) using LangGraph. Data flows between nodes via a strictly typed `PatientState`. The workflow consists of four primary nodes:

1. **Router:** The entry point. It inspects the patient's data profile to detect which modalities are available and which are missing.
2. **Planner:** Acts as the reasoning engine. It analyzes the patient's available clinical features and formulates a strategy for imputation and survival prediction.
3. **Generator:** Responsible for producing plausible feature arrays for missing modalities. Development progresses from statistical baselines (e.g., mean imputation via cosine similarity) to advanced generative modeling.
4. **Verifier:** Assesses the quality and biological plausibility of the generated data. It implements a self-refinement loop: if the generated data fails the verification threshold, the Verifier provides correction hints and routes the execution back to the Generator.

## Repository Structure
The project is organized to separate data ingestion, multi-agent orchestration, and evaluation:

```text
project/
├── data/                  # IGNORED BY GIT - Local storage for dataset files (cache_data.zip, .pkl, .json)
├── notebooks/             # Jupyter notebooks for Exploratory Data Analysis (EDA) and prototyping
├── scripts/               # Utility scripts (e.g., graph compilation and smoke tests)
├── src/
│   ├── baseline/          # Machine Learning baseline models for survival prediction comparison
│   ├── orchestrator/      # LangGraph multi-agent system implementation
│   │   ├── nodes/         # Individual node logic (router.py, planner.py, generator.py, verifier.py)
│   │   ├── graph.py       # DAG definition and conditional routing logic
│   │   └── state.py       # TypedDict defining the shared PatientState contract
│   └── data_loader.py     # Multimodal data ingestion, shape validation, and split management
├── tests/                 # Integration and unit tests (Pytest)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/MateCle/agentic-multimodal-lung-cancer-ai.git
cd agentic-multimodal-lung-cancer-ai
```

### 2. Python Environment Setup
You can use either standard Python `venv` or `conda`. Python 3.10+ is recommended.

**Option A: Using Python venv (Standard)**
*On Windows:*
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
*On macOS/Linux:*
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Option B: Using Conda**
```bash
conda create -n amlc python=3.10
conda activate amlc
pip install -r requirements.txt
```

### 3. Data Placement (CRITICAL)
Dataset files are strictly excluded from version control to prevent repository bloat and comply with data handling practices.
Ensure that your raw data files (`cache_data.zip` and the extracted `.pkl` / `.json` files) are placed explicitly inside the `data/` directory. 

## Testing
The project uses `pytest` for integration and unit testing. Tests ensure the reliability of the data loader against actual data structures and validate the expected behavior of the LangGraph state.
```bash
pytest tests/ -v
```
