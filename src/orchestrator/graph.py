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
    LanguageAgent,
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
_LANG_PARSER = "language_parser"
_LANG_REPORTER = "language_reporter"


def _load_metadata_per_cohort(data_dir: Path) -> dict:
    """Load feature-name metadata for both cohorts, keyed by cohort."""
    out: dict = {}
    for cohort in ("luad", "lusc"):
        meta_path = data_dir / f"tcga_{cohort}_metadata.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                out[cohort] = pickle.load(f)
        else:
            out[cohort] = {}
    return out


def _load_metadata(data_dir: Path) -> dict:
    """Backward-compatible: return metadata of the first cohort found.

    Used by the Miner / agent prompts where transcriptomics column names
    are shared across cohorts.
    """
    per_cohort = _load_metadata_per_cohort(data_dir)
    for cohort in ("luad", "lusc"):
        if per_cohort.get(cohort):
            return per_cohort[cohort]
    return {}


def _make_language_parser_node(language_agent: LanguageAgent):
    """Entry node: parse user query into a structured patient ID."""

    def language_parser_node(state: PatientState) -> dict:
        raw_query = state.get("user_query", "")

        # If the user already provided a patient_id directly (CLI --patient),
        # skip parsing and use it as-is.
        existing_pid = state.get("patient_id", "")
        if existing_pid and not raw_query:
            return {
                "parsed_query": {
                    "patient_id": existing_pid,
                    "cohort": None,
                    "raw_query": "",
                    "error": None,
                },
                "execution_log": [
                    f"[LanguageAgent] Direct ID input: {existing_pid}. "
                    f"Skipping query parsing."
                ],
            }

        parsed = language_agent.parse_query(raw_query)

        log_lines = [
            f"[LanguageAgent] Query: '{raw_query[:80]}{'...' if len(raw_query) > 80 else ''}'"
        ]
        if parsed.error:
            log_lines.append(f"[LanguageAgent] Parse error: {parsed.error}")
            return {
                "parsed_query": parsed.to_dict(),
                "clinical_report": (
                    f"# Error\n\n{parsed.error}\n\nOriginal query: `{raw_query}`"
                ),
                "execution_log": log_lines,
            }

        log_lines.append(
            f"[LanguageAgent] Parsed: patient_id={parsed.patient_id}, "
            f"cohort={parsed.cohort}"
        )
        return {
            "patient_id": parsed.patient_id,
            "parsed_query": parsed.to_dict(),
            "execution_log": log_lines,
        }

    return language_parser_node


def _make_language_reporter_node(language_agent: LanguageAgent):
    """Exit node: generate the markdown clinical report from final state."""

    def language_reporter_node(state: PatientState) -> dict:
        # If the parser short-circuited the graph due to a parsing error,
        # it already set clinical_report to an error message. Don't overwrite.
        existing = state.get("clinical_report", "")
        parse_failed = bool(state.get("parsed_query", {}).get("error"))
        if parse_failed and existing:
            return {
                "execution_log": [
                    "[LanguageAgent] Parse-error report preserved; "
                    "skipping full report generation."
                ],
            }

        report = language_agent.generate_report(state)
        return {
            "clinical_report": report,
            "execution_log": [
                f"[LanguageAgent] Generated clinical report ({len(report)} chars)."
            ],
        }

    return language_reporter_node


def route_after_language_parser(state: PatientState) -> str:
    """Skip the rest of the graph if parsing failed."""
    parsed = state.get("parsed_query", {})
    if parsed.get("error") or not parsed.get("patient_id"):
        return _LANG_REPORTER  # straight to exit, will report the error
    return _DATA_LOADER


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
            "risk_class": "",
            "top_shap_features": [],
            "source_map": {},
            "clinical_report": "",
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
    model_name: str = "coxnet",
    imputation: str = "mice",
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
    metadata_per_cohort = _load_metadata_per_cohort(data_dir)

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

    # LanguageAgent: entry (parse query) + exit (generate report)
    language_agent = LanguageAgent(llm)
    builder.add_node(_LANG_PARSER, _make_language_parser_node(language_agent))
    builder.add_node(_LANG_REPORTER, _make_language_reporter_node(language_agent))

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

    # Predictor: baseline pipeline + per-cohort metadata for SHAP, or mock
    if pipelines:
        builder.add_node(
            _PREDICTOR,
            make_predictor_node(pipelines, metadata_per_cohort),
        )
    else:
        builder.add_node(_PREDICTOR, mock_predictor)

    # --- Edges ---
    builder.set_entry_point(_LANG_PARSER)

    # LanguageAgent parser -> DataLoader (or straight to reporter on error)
    builder.add_conditional_edges(
        _LANG_PARSER,
        route_after_language_parser,
        {_DATA_LOADER: _DATA_LOADER, _LANG_REPORTER: _LANG_REPORTER},
    )

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

    # Predictor -> LanguageAgent reporter (exit)
    builder.add_edge(_PREDICTOR, _LANG_REPORTER)
    builder.add_edge(_LANG_REPORTER, END)

    return builder.compile()
