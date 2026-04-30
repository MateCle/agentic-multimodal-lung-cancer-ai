"""
LangGraph DAG definition for the multimodal lung cancer orchestrator.
AFM2-aligned core pipeline:

  DataLoader -> Planner -> Miner (LLM) -> Generator (k-NN) -> Verifier (LLM) -> Predictor
                       |                                          ^
                       | (all present)                            | (self-refinement, max 3)
                       +---> Predictor                            v
                                                              Generator

Extensions (added after core works):
    - Modality agents (parallel on multi-GPU)
    - TCGA MMKG retrieval
    - SPOKE verification
    - FAISS GPU acceleration
"""

import logging
import pickle
from pathlib import Path

from langgraph.graph import END, StateGraph

from src.baseline.pipeline import load_pipeline, pipeline_path
from src.data_loader import MODALITY_DIMS, load_patient, load_raw_data
from src.orchestrator.agents import (
    ClinicalAgent,
    GenomicAgent,
    MethylationAgent,
    VisualAgent,
)
from src.orchestrator.agents.clinical import infer_clinical_column_types
from src.orchestrator.llm import get_llm_client
from src.orchestrator.nodes.generator import (
    build_pool_index,
    make_generator_node,
)
from src.orchestrator.nodes.generator import (
    generator_node as mock_generator,
)

# Core nodes (real + mock)
from src.orchestrator.nodes.miner import make_miner_node
from src.orchestrator.nodes.miner import miner_node as mock_miner
from src.orchestrator.nodes.planner import planner_node
from src.orchestrator.nodes.predictor import (
    make_predictor_node,
)
from src.orchestrator.nodes.predictor import (
    predictor_node as mock_predictor,
)
from src.orchestrator.nodes.router import route_after_planner, route_after_verifier
from src.orchestrator.nodes.verifier import (
    build_pool_stats,
    make_verifier_node,
)
from src.orchestrator.nodes.verifier import (
    verifier_node as mock_verifier,
)
from src.orchestrator.state import PatientState

logger = logging.getLogger(__name__)

_DATA_LOADER = "data_loader"
_PLANNER = "planner"
_MINER = "miner"
_GENERATOR = "generator"
_VERIFIER = "verifier"
_PREDICTOR = "predictor"


def _load_metadata(data_dir: Path) -> dict:
    """Load feature name metadata from either cohort (columns are shared for transcriptomics)."""
    for cohort in ("luad", "lusc"):
        meta_path = data_dir / f"tcga_{cohort}_metadata.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                return pickle.load(f)
    return {}


def _make_data_loader_node(all_data: dict, cohort_map: dict):
    """Returns a data_loader closure."""

    def data_loader_node(state: PatientState) -> dict:
        pid = state["patient_id"]
        record = load_patient(pid, all_data)

        if record is None:
            raise ValueError(f"[DataLoader] Patient '{pid}' not found.")

        cohort = cohort_map.get(pid, "unknown")
        log = (
            f"[DataLoader] Loaded {pid} (cohort={cohort.upper()}). "
            f"Available: {record['available_modalities']}. "
            f"Missing: {record['missing_modalities']}."
        )
        return {
            "cohort": cohort,
            "clinical": record["clinical"],
            "transcriptomics": record["transcriptomics"],
            "wsi": record["wsi"],
            "methylation": record["methylation"],
            "available_modalities": record["available_modalities"],
            "missing_modalities": record["missing_modalities"],
            "agent_summaries": {},
            "mining_rules": {},
            "generated_modalities": {},
            "verification_scores": {},
            "verification_passed": False,
            "survival_prediction": None,
            "routing_decision": "",
            "execution_log": [log],
        }

    return data_loader_node


def _load_pipelines(model_name: str, imputation: str) -> dict:
    """Load fitted baseline pipelines for both cohorts."""
    pipelines = {}
    for cohort in ("luad", "lusc"):
        path = pipeline_path(cohort, model_name, imputation)
        if path.exists():
            pipelines[cohort] = load_pipeline(cohort, model_name, imputation)
        else:
            logger.warning(f"No pipeline for {cohort}/{model_name}/{imputation}.")
    return pipelines


def build_graph(
    data_dir: Path,
    model_name: str = "coxph",
    imputation: str = "zero",
    train_patient_ids: list[str] | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
):
    """
    Build and compile the LangGraph orchestrator.

    Modes (auto-detected):
        - Full AFM2: train IDs + LLM → real Miner, Generator (k-NN), Verifier
        - Mock: no train IDs → all placeholder nodes

    Args:
        data_dir:          Path to cache_data directory.
        model_name:        Baseline model for the Predictor.
        imputation:        Imputation strategy the baseline was trained with.
        train_patient_ids: Patient IDs for the retrieval pool.
        llm_provider:      Override LLM provider ('openai', 'mock').
        llm_model:         Override LLM model name.

    Returns:
        Compiled LangGraph runnable.
    """
    data_dir = Path(data_dir)

    # --- Load data ---
    luad_data, _ = load_raw_data(data_dir, "luad")
    lusc_data, _ = load_raw_data(data_dir, "lusc")
    all_data = {**luad_data, **lusc_data}

    cohort_map = dict.fromkeys(luad_data, "luad")
    cohort_map.update(dict.fromkeys(lusc_data, "lusc"))

    # --- Load metadata (feature names for Miner prompts) ---
    metadata = _load_metadata(data_dir)

    # --- Load baseline pipelines ---
    pipelines = _load_pipelines(model_name, imputation)

    # --- Initialize pool + LLM ---
    pool = None
    pool_stats = None
    llm = None
    clinical_column_types: list[str] = []

    if train_patient_ids is not None:
        pool = build_pool_index(all_data, train_patient_ids)
        pool_stats = build_pool_stats(pool)
        clinical_column_types = infer_clinical_column_types(
            pool, MODALITY_DIMS["clinical"]
        )
        llm = get_llm_client(provider=llm_provider, model=llm_model)

        logger.info(
            f"AFM2 mode: pool={len(pool)} patients, LLM={llm.__class__.__name__}"
        )

    # --- Build graph ---
    builder = StateGraph(PatientState)

    # DataLoader + Planner (always real)
    builder.add_node(_DATA_LOADER, _make_data_loader_node(all_data, cohort_map))
    builder.add_node(_PLANNER, planner_node)

    # Miner: two-stage with parallel modality agents, or mock
    if llm is not None:
        agents = {
            "clinical": ClinicalAgent(
                llm, metadata, clinical_column_types=clinical_column_types
            ),
            "transcriptomics": GenomicAgent(llm, metadata),
            "wsi": VisualAgent(
                llm,
                metadata,
                pool=pool,
                n_neighbors=3,
                clinical_column_types=clinical_column_types,
            ),
            "methylation": MethylationAgent(llm, metadata),
        }
        builder.add_node(_MINER, make_miner_node(llm, agents))
    else:
        builder.add_node(_MINER, mock_miner)

    # Generator: LLM-guided k-NN retrieval with pool, or mock
    if pool is not None:
        builder.add_node(_GENERATOR, make_generator_node(pool, llm, metadata))
    else:
        builder.add_node(_GENERATOR, mock_generator)

    # Verifier: distributional + LLM scoring, or mock
    if pool_stats is not None and llm is not None:
        builder.add_node(_VERIFIER, make_verifier_node(pool_stats, llm))
    else:
        builder.add_node(_VERIFIER, mock_verifier)

    # Predictor: baseline pipeline, or mock
    if pipelines:
        builder.add_node(_PREDICTOR, make_predictor_node(pipelines))
    else:
        builder.add_node(_PREDICTOR, mock_predictor)

    # --- Edges ---
    builder.set_entry_point(_DATA_LOADER)
    builder.add_edge(_DATA_LOADER, _PLANNER)

    builder.add_conditional_edges(
        _PLANNER,
        route_after_planner,
        {_MINER: _MINER, _PREDICTOR: _PREDICTOR},
    )

    builder.add_edge(_MINER, _GENERATOR)
    builder.add_edge(_GENERATOR, _VERIFIER)

    builder.add_conditional_edges(
        _VERIFIER,
        route_after_verifier,
        {_PREDICTOR: _PREDICTOR, _GENERATOR: _GENERATOR},
    )

    builder.add_edge(_PREDICTOR, END)

    return builder.compile()
