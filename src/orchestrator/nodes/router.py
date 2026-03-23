"""
Conditional routing functions for the LangGraph orchestrator.
These are NOT nodes — they are passed to add_conditional_edges() in graph.py.
"""
from typing import Literal
from src.orchestrator.state import PatientState

MAX_REFINEMENT_ATTEMPTS = 3


def route_after_planner(state: PatientState) -> Literal["miner", "predictor"]:
    """
    After Planner:
    - Missing modalities present -> Miner (which feeds into Generator).
    - All modalities present     -> skip directly to Predictor.
    """
    if state["routing_decision"] == "generate":
        return "miner"
    return "predictor"


def route_after_verifier(state: PatientState) -> Literal["predictor", "generator"]:
    """
    Self-refinement loop after Verifier:
    - Passed or attempt limit reached -> Predictor.
    - Failed and under limit          -> Generator (Miner rules are reused).
    """
    attempts = sum(
        1 for line in state["execution_log"] if "[Generator]" in line
    )
    if state["verification_passed"] or attempts >= MAX_REFINEMENT_ATTEMPTS:
        return "predictor"
    return "generator"
