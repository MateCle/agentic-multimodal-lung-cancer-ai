"""
Predictor node for the LangGraph orchestrator.
Assembles all features (real + generated) and runs the fitted
baseline pipeline (scaler → PCA → survival model) for prediction.

Usage in graph.py:
    from src.orchestrator.nodes.predictor import make_predictor_node
    predictor = make_predictor_node(pipelines)
    builder.add_node("predictor", predictor)
"""

import random

import numpy as np

from src.data_loader import MODALITY_DIMS, MODALITY_KEYS
from src.orchestrator.state import PatientState


def _assemble_features(state: PatientState) -> tuple[np.ndarray, list[str]]:
    """
    Assemble a single feature vector from real + generated modalities.
    Returns (feature_vector, source_list) where source_list tracks
    where each modality came from ('real', 'generated', or 'zero').
    """
    generated = state.get("generated_modalities") or {}
    vectors = []
    sources = []

    for mod in MODALITY_KEYS:
        dim = MODALITY_DIMS[mod]
        real_data = state.get(mod)

        if real_data is not None:
            arr = np.array(real_data).flatten().astype(np.float64)
            if arr.size == dim:
                vectors.append(arr)
                sources.append(f"{mod}:real")
                continue

        if mod in generated:
            arr = np.array(generated[mod]).flatten().astype(np.float64)
            if arr.size == dim:
                vectors.append(arr)
                sources.append(f"{mod}:generated")
                continue

        vectors.append(np.zeros(dim, dtype=np.float64))
        sources.append(f"{mod}:zero")

    return np.concatenate(vectors), sources


def make_predictor_node(pipelines: dict):
    """
    Returns a Predictor closure over fitted baseline pipelines.

    Args:
        pipelines: Dict mapping cohort name to
                   {"model": fitted_model, "scaler": fitted_scaler, "pca": fitted_pca}
    """

    def predictor_node(state: PatientState) -> dict:
        cohort = state.get("cohort", "unknown")
        features, sources = _assemble_features(state)

        pipeline = pipelines.get(cohort)
        if pipeline is None:
            prediction = round(random.uniform(0.0, 1.0), 4)
            log = (
                f"[Predictor] No pipeline for cohort '{cohort}'. "
                f"Random fallback: {prediction:.4f}. "
                f"Sources: {sources}"
            )
            return {"survival_prediction": prediction, "execution_log": [log]}

        model = pipeline.model
        scaler = pipeline.scaler
        pca = pipeline.pca

        X = features.reshape(1, -1)

        # Handle dimensionality: MICE produces ~200 dims, others 19077
        expected_features = scaler.n_features_in_
        if X.shape[1] != expected_features:
            # Pad or truncate to match scaler expectation
            if X.shape[1] < expected_features:
                X = np.pad(X, ((0, 0), (0, expected_features - X.shape[1])))
            else:
                X = X[:, :expected_features]

        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        risk_score = float(model.predict_risk(X_pca)[0])

        log = (
            f"[Predictor] Cohort={cohort.upper()}, "
            f"model={model.__class__.__name__}, "
            f"risk_score={risk_score:.4f}. "
            f"Sources: {sources}"
        )

        return {"survival_prediction": risk_score, "execution_log": [log]}

    return predictor_node


def predictor_node(state: PatientState) -> dict:  # NOSONAR
    """MOCK fallback: random survival prediction."""
    prediction = round(random.uniform(0.0, 1.0), 4)
    log = f"[Predictor] MOCK survival prediction (DSS): {prediction:.4f}."
    return {"survival_prediction": prediction, "execution_log": [log]}
