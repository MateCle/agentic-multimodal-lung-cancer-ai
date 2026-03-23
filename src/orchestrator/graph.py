"""
LangGraph DAG definition for the multimodal lung cancer orchestrator.
Implements the AFM2-inspired pipeline:

  DataLoader -> Planner -> Miner -> Generator -> Verifier -> Predictor
                       |                              ^
                       | (all modalities present)     | (self-refinement loop)
                       +------------------------------+-> Predictor
"""
from pathlib import Path

from langgraph.graph import StateGraph, END

from src.orchestrator.state import PatientState
from src.orchestrator.nodes import (
    planner_node,
    miner_node,
    generator_node,
    verifier_node,
    predictor_node,
    route_after_planner,
    route_after_verifier,
)
from src.data_loader import load_raw_data, load_patient

_DATA_LOADER = "data_loader"
_PLANNER     = "planner"
_MINER       = "miner"
_GENERATOR   = "generator"
_VERIFIER    = "verifier"
_PREDICTOR   = "predictor"


def _make_data_loader_node(all_data: dict):
    """
    Returns a data_loader closure over the preloaded cohort dict.
    PKL files are read once at build_graph() time, not per invocation.
    """
    def data_loader_node(state: PatientState) -> dict:
        pid    = state["patient_id"]
        record = load_patient(pid, all_data)

        if record is None:
            raise ValueError(
                f"[DataLoader] Patient '{pid}' not found in cohort data."
            )

        log = (
            f"[DataLoader] Loaded {pid}. "
            f"Available: {record['available_modalities']}. "
            f"Missing: {record['missing_modalities']}."
        )
        return {
            "clinical":             record["clinical"],
            "transcriptomics":      record["transcriptomics"],
            "wsi":                  record["wsi"],
            "methylation":          record["methylation"],
            "available_modalities": record["available_modalities"],
            "missing_modalities":   record["missing_modalities"],
            "mining_rules":         {},
            "generated_modalities": {},
            "verification_scores":  {},
            "verification_passed":  False,
            "survival_prediction":  None,
            "routing_decision":     "",
            "execution_log":        [log],
        }

    return data_loader_node


def build_graph(data_dir: Path):
    """
    Loads cohort data, builds and compiles the LangGraph StateGraph.

    Args:
        data_dir: Path to the cache_data directory (contains .pkl files).

    Returns:
        A compiled LangGraph runnable ready to be called with .invoke().
    """
    data_dir = Path(data_dir)

    luad_data, _ = load_raw_data(data_dir, "luad")
    lusc_data, _ = load_raw_data(data_dir, "lusc")
    all_data      = {**luad_data, **lusc_data}

    builder = StateGraph(PatientState)

    builder.add_node(_DATA_LOADER, _make_data_loader_node(all_data))
    builder.add_node(_PLANNER,     planner_node)
    builder.add_node(_MINER,       miner_node)
    builder.add_node(_GENERATOR,   generator_node)
    builder.add_node(_VERIFIER,    verifier_node)
    builder.add_node(_PREDICTOR,   predictor_node)

    builder.set_entry_point(_DATA_LOADER)

    builder.add_edge(_DATA_LOADER, _PLANNER)

    # Planner: missing modalities -> Miner, all present -> Predictor
    builder.add_conditional_edges(
        _PLANNER,
        route_after_planner,
        {_MINER: _MINER, _PREDICTOR: _PREDICTOR},
    )

    # Miner always feeds into Generator
    builder.add_edge(_MINER, _GENERATOR)

    # Generator always feeds into Verifier
    builder.add_edge(_GENERATOR, _VERIFIER)

    # Verifier: pass -> Predictor, fail -> Generator (self-refinement)
    builder.add_conditional_edges(
        _VERIFIER,
        route_after_verifier,
        {_PREDICTOR: _PREDICTOR, _GENERATOR: _GENERATOR},
    )

    builder.add_edge(_PREDICTOR, END)

    return builder.compile()
