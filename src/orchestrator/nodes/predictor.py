"""
Predictor node for the LangGraph orchestrator.
MOCK: returns a random DSS survival prediction.
"""
import random
from src.orchestrator.state import PatientState


def predictor_node(state: PatientState) -> dict:
    prediction = round(random.uniform(0.0, 1.0), 4)
    log        = f"[Predictor] Survival prediction (DSS): {prediction:.4f}."
    return {"survival_prediction": prediction, "execution_log": [log]}
