"""
Verifier node for the LangGraph orchestrator.
MOCK: assigns a random quality score to each generated modality.
"""
import random
from src.orchestrator.state import PatientState

VERIFIER_PASS_THRESHOLD = 0.5


def verifier_node(state: PatientState) -> dict:
    generated = state.get("generated_modalities") or {}
    scores    = {}
    log_lines = []

    for modality in generated:
        score            = round(random.uniform(0.0, 1.0), 4)
        scores[modality] = score
        log_lines.append(
            f"[Verifier] '{modality}' score: {score:.4f} "
            f"(threshold={VERIFIER_PASS_THRESHOLD})."
        )

    passed = all(s >= VERIFIER_PASS_THRESHOLD for s in scores.values())
    log_lines.append(f"[Verifier] Overall: {'PASS' if passed else 'FAIL'}.")

    return {
        "verification_scores": scores,
        "verification_passed": passed,
        "execution_log":       log_lines,
    }
