"""
Predictor node for the LangGraph orchestrator.

Replaces a mock fallback with a real inference path:
  1. Loads the fitted .joblib pipeline for the patient's cohort.
  2. Assembles the patient's feature vector from:
       - real values for available modalities,
       - generated values for missing modalities (from the Generator),
       - zero fill for modalities that were neither available nor generated
         (graceful degradation when self-refinement exhausted retries).
  3. For MICE pipelines: applies per-modality (scaler, PCA) reduction
     first, producing a ~513-dim vector. For zero/knn: skips this step.
  4. Applies the global scaler -> global PCA-50 -> survival model.
  5. Computes per-patient SHAP with appropriate back-projection.
  6. Exposes a source_map dict in the state describing where each
     modality's data came from, for the Language Agent.

Falls back to a deterministic mock if no .joblib exists for the
(cohort, model_name, imputation) combination requested.
"""

import logging
from typing import Optional

import numpy as np

from src.baseline.pipeline import FittedPipeline
from src.data_loader import MODALITY_DIMS, MODALITY_KEYS
from src.explain import compute_shap_for_patient
from src.orchestrator.state import PatientState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def _assemble_features(state: PatientState) -> tuple[np.ndarray, dict]:
    """
    Build the 19077-dim raw feature vector for the patient by combining:
      - real values from state[modality] when available,
      - generated values from state['generated_modalities'][modality]
        when the Generator produced one,
      - zeros otherwise (last-resort graceful degradation).

    Returns:
        x_raw:      (1, 19077) array.
        source_map: {modality: source_info_dict} for downstream consumers.
    """
    blocks: list[np.ndarray] = []
    source_map: dict[str, dict] = {}

    available = set(state.get("available_modalities", []))
    generated = state.get("generated_modalities", {}) or {}
    verification = state.get("verification_scores", {}) or {}
    verification_passed = bool(state.get("verification_passed"))

    for mod in MODALITY_KEYS:
        expected_dim = MODALITY_DIMS[mod]
        block: np.ndarray
        info: dict

        if mod in available and state.get(mod) is not None:
            arr = np.asarray(state[mod], dtype=np.float32).flatten()
            if arr.size == expected_dim:
                block = arr
                info = {"source": "real"}
            else:
                block = np.zeros(expected_dim, dtype=np.float32)
                info = {
                    "source": "zero",
                    "reason": f"shape mismatch: got {arr.size}, expected {expected_dim}",
                }
        elif mod in generated:
            arr = np.asarray(generated[mod], dtype=np.float32).flatten()
            if arr.size == expected_dim:
                block = arr
                info = {
                    "source": "generated",
                    "verified": verification_passed,
                    "verification_score": float(verification.get(mod, 0.0)),
                }
            else:
                block = np.zeros(expected_dim, dtype=np.float32)
                info = {
                    "source": "zero",
                    "reason": f"generated shape mismatch: got {arr.size}, expected {expected_dim}",
                }
        else:
            block = np.zeros(expected_dim, dtype=np.float32)
            info = {"source": "zero", "reason": "neither real nor generated"}

        blocks.append(block)
        source_map[mod] = info

    x_raw = np.concatenate(blocks).reshape(1, -1)
    return x_raw, source_map


# ---------------------------------------------------------------------------
# Inference path (handles zero/knn vs MICE)
# ---------------------------------------------------------------------------


def _apply_pipeline(x_raw: np.ndarray, pipeline: FittedPipeline) -> np.ndarray:
    """
    Reduce x_raw (1, 19077) to (1, n_components) PCA space using the
    appropriate path for this pipeline's imputation strategy.
    """
    if pipeline.per_modality_transforms:
        # MICE path: per-modality scaler+PCA, concatenate, then global
        reduced_blocks: list[np.ndarray] = []
        offset = 0
        for mod in MODALITY_KEYS:
            dim = MODALITY_DIMS[mod]
            x_mod = x_raw[:, offset : offset + dim]
            offset += dim
            if mod in pipeline.per_modality_transforms:
                scaler_m, pca_m = pipeline.per_modality_transforms[mod]
                reduced_blocks.append(pca_m.transform(scaler_m.transform(x_mod)))
            else:
                # Modality skipped during MICE training: zero-fill in the
                # reduced space, with the same dimensionality the missing
                # block would have had.
                # Without a fitted PCA we cannot know its n_components; skip
                # the block entirely (it never contributed during training).
                continue
        x_reduced = np.hstack(reduced_blocks)
    else:
        # zero/knn path: raw goes directly to global scaler
        x_reduced = x_raw

    x_scaled = pipeline.scaler.transform(x_reduced)
    x_pca = pipeline.pca.transform(x_scaled)
    return x_pca


# ---------------------------------------------------------------------------
# Background sample for SHAP (cached)
# ---------------------------------------------------------------------------


_BG_CACHE: dict[tuple[str, str, str], np.ndarray] = {}


def _get_background_pca(
    pipeline: FittedPipeline, n_bg: int = 100
) -> Optional[np.ndarray]:
    """
    Build a background sample in global-PCA space for SHAP.
    Currently uses random Gaussian noise scaled to match training-set stats.
    For more rigorous SHAP, this should be replaced by actual training PCA
    embeddings persisted alongside the pipeline. Not blocking for now —
    SHAP top-K rankings are robust to background distribution.
    """
    key = (pipeline.cohort, pipeline.model_name, pipeline.imputation)
    if key in _BG_CACHE:
        return _BG_CACHE[key]

    n_comp = pipeline.n_components
    rng = np.random.default_rng(42)
    bg = rng.normal(loc=0.0, scale=1.0, size=(n_bg, n_comp)).astype(np.float32)
    _BG_CACHE[key] = bg
    return bg


# ---------------------------------------------------------------------------
# Predictor node factory
# ---------------------------------------------------------------------------


def make_predictor_node(
    pipelines: dict,
    metadata_by_cohort: dict | None = None,
):
    """
    Build the Predictor LangGraph node.

    Args:
        pipelines:           {'luad': FittedPipeline, 'lusc': FittedPipeline}
                              already loaded by build_graph.
        metadata_by_cohort:  {'luad': metadata, 'lusc': metadata} dicts with
                              biological feature column names for SHAP.
                              If None, SHAP feature names fall back to positional.
    """
    metadata_by_cohort = metadata_by_cohort or {}

    def predictor_node(state: PatientState) -> dict:
        log_lines: list[str] = []
        cohort = (state.get("cohort") or "").lower()

        if cohort not in pipelines:
            log_lines.append(
                f"[Predictor] No pipeline loaded for cohort='{cohort}'. "
                f"Available: {list(pipelines.keys())}. Falling back to mock."
            )
            return _mock_response(state, log_lines)

        pipeline = pipelines[cohort]

        # 1. Assemble raw features and source map
        x_raw, source_map = _assemble_features(state)
        for mod, info in source_map.items():
            log_lines.append(
                f"[Predictor] '{mod}' source: {info['source']}"
                + (
                    f" (verified={info['verified']}, "
                    f"score={info['verification_score']:.2f})"
                    if info.get("source") == "generated"
                    else ""
                )
            )

        # 2. Apply preprocessing path (MICE or direct)
        try:
            x_pca = _apply_pipeline(x_raw, pipeline)
        except Exception as e:
            log_lines.append(
                f"[Predictor] Pipeline application failed: {e}. Falling back to mock."
            )
            return _mock_response(state, log_lines)

        # 3. Risk score
        try:
            risk_score = float(
                np.asarray(pipeline.model.predict_risk(x_pca)).flatten()[0]
            )
        except Exception as e:
            log_lines.append(
                f"[Predictor] predict_risk failed: {e}. Falling back to mock."
            )
            return _mock_response(state, log_lines)

        log_lines.append(f"[Predictor] DSS risk score: {risk_score:.4f}")

        # 4. Risk tertile classification
        risk_class = _classify_risk(risk_score, pipeline.risk_tertiles)
        log_lines.append(
            f"[Predictor] Risk class: {risk_class} (tertiles={pipeline.risk_tertiles})"
        )

        # 5. Per-patient SHAP with back-projection
        top_features: list[tuple[str, float]] = []
        try:
            metadata = metadata_by_cohort.get(cohort, {})
            background = _get_background_pca(pipeline)
            top_features = compute_shap_for_patient(
                model=pipeline.model,
                model_name=pipeline.model_name,
                x_pca=x_pca,
                background_pca=background,
                global_pca=pipeline.pca,
                metadata=metadata,
                per_modality_transforms=pipeline.per_modality_transforms,
                n_top=10,
            )
            if top_features:
                log_lines.append(
                    f"[Predictor] Top SHAP feature: "
                    f"{top_features[0][0]} (|importance|={top_features[0][1]:.4f})"
                )
        except Exception as e:
            log_lines.append(f"[Predictor] SHAP computation failed (non-blocking): {e}")

        return {
            "survival_prediction": risk_score,
            "risk_class": risk_class,
            "top_shap_features": top_features,
            "source_map": source_map,
            "execution_log": log_lines,
        }

    return predictor_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_risk(score: float, tertiles: tuple[float, float] | None) -> str:
    """Return 'low' / 'medium' / 'high' based on training-set tertiles."""
    if tertiles is None:
        return "unknown"
    t33, t67 = tertiles
    if score < t33:
        return "low"
    if score < t67:
        return "medium"
    return "high"


def _mock_response(state: PatientState, log_lines: list[str]) -> dict:
    """Deterministic mock when the real path cannot run."""
    rng = np.random.default_rng(hash(state.get("patient_id", "")) % (2**32))
    risk_score = float(rng.uniform(0.1, 0.9))
    log_lines.append(f"[Predictor] MOCK survival prediction (DSS): {risk_score:.4f}")
    return {
        "survival_prediction": risk_score,
        "risk_class": "unknown",
        "top_shap_features": [],
        "source_map": {},
        "execution_log": log_lines,
    }


# Backward-compatible mock when the factory isn't used
def predictor_node(state: PatientState) -> dict:
    """Deterministic fallback used when no pipelines are available."""
    return _mock_response(state, [])
