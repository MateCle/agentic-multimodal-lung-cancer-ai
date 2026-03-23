"""
Planner node for the LangGraph orchestrator.
Decides routing strategy based on missing modalities — no LLM calls.
"""
from src.orchestrator.state import PatientState


def planner_node(state: PatientState) -> dict:
    missing  = state["missing_modalities"]
    decision = "generate" if missing else "predict"
    log = (
        f"[Planner] Missing modalities: {missing}. "
        f"Routing decision: '{decision}'."
    )
    return {"routing_decision": decision, "execution_log": [log]}
