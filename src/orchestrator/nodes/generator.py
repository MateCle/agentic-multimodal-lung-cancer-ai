"""
Generator node for the LangGraph orchestrator.
Based on AFM2: receives mining rules from the Miner and generates
imputed arrays for each missing modality.

MOCK constraint: outputs zero arrays regardless of the rules.
In production, use the rule string to condition a modality-specific model.
"""
import numpy as np
from src.data_loader import MODALITY_DIMS
from src.orchestrator.state import PatientState


def generator_node(state: PatientState) -> dict:
    """
    For each missing modality, creates a zero-filled numpy array.
    The mining_rules from the Miner are logged to confirm they are received.
    In production they would condition the generative model.

    MOCK: no LLM calls, no learned model — zeros only.
    """
    missing   = state["missing_modalities"]
    rules     = state.get("mining_rules") or {}
    generated = dict(state.get("generated_modalities") or {})
    log_lines = []

    for modality in missing:
        dim                 = MODALITY_DIMS[modality]
        generated[modality] = np.zeros(dim)
        rule_used           = rules.get(modality, "no rule available")
        log_lines.append(
            f"[Generator] Imputed '{modality}' with zeros, shape=({dim},). "
            f"Rule applied: '{rule_used[:60]}...'"
        )

    return {"generated_modalities": generated, "execution_log": log_lines}
