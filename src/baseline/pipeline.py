"""
Pipeline serialization utilities for the baseline survival models.
Saves and loads the fitted (model, scaler, PCA) triple so that the
Predictor node in the orchestrator can reuse the trained pipeline
without re-fitting.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PIPELINE_DIR = Path("models/baseline")


@dataclass
class FittedPipeline:
    """Container for a fitted baseline pipeline."""

    model: object  # CoxPHBaseline | XGBoostSurvivalModel | ...
    scaler: StandardScaler
    pca: PCA
    model_name: str
    imputation: str
    cohort: str
    n_components: int
    explained_variance: float
    # MICE-only: per-modality (scaler, PCA) pairs for inference-time reduction.
    # None for zero/knn/knn_tuned where raw features go directly to global scaler.
    per_modality_transforms: dict | None = None
    # Risk score tertile thresholds computed on training set, for clinical
    # stratification (low/medium/high) by the Language Agent at inference time.
    risk_tertiles: tuple[float, float] | None = None


def pipeline_path(cohort: str, model_name: str, imputation: str) -> Path:
    """Deterministic path for a given (cohort, model, imputation) combo."""
    return PIPELINE_DIR / f"pipeline_{cohort}_{model_name}_{imputation}.joblib"


def save_pipeline(
    model: object,
    scaler: StandardScaler,
    pca: PCA,
    cohort: str,
    model_name: str,
    imputation: str,
    per_modality_transforms: dict | None = None,
    risk_tertiles: tuple[float, float] | None = None,
) -> Path:
    """
    Serialize a fitted pipeline to disk.

    Args:
        model:       Fitted survival model with .predict_risk() method.
        scaler:      Fitted StandardScaler (global, post-imputation).
        pca:         Fitted PCA transformer (global, 50 components).
        cohort:      'luad' or 'lusc'.
        model_name:  'coxph', 'xgboost', etc.
        imputation:  'zero', 'knn', 'knn_tuned', 'mice'.
        per_modality_transforms: For MICE only: dict[str, (scaler, PCA)] with
                     fitted per-modality transforms. None for other strategies.
        risk_tertiles: Tuple (threshold_33, threshold_67) of training risk
                     scores, used by the Language Agent for low/medium/high
                     classification.

    Returns:
        Path to the saved .joblib file.
    """
    pipe = FittedPipeline(
        model=model,
        scaler=scaler,
        pca=pca,
        model_name=model_name,
        imputation=imputation,
        cohort=cohort,
        n_components=pca.n_components_,
        explained_variance=float(pca.explained_variance_ratio_.sum()),
        per_modality_transforms=per_modality_transforms,
        risk_tertiles=risk_tertiles,
    )

    path = pipeline_path(cohort, model_name, imputation)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    print(f"  [Pipeline] Saved to {path}")
    return path


def load_pipeline(
    cohort: str,
    model_name: str,
    imputation: str,
) -> FittedPipeline:
    """
    Load a previously saved pipeline.

    Raises:
        FileNotFoundError if the pipeline hasn't been trained yet.
    """
    path = pipeline_path(cohort, model_name, imputation)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved pipeline at {path}. "
            f"Run main_baseline.py with --model {model_name} "
            f"--imputation {imputation} first."
        )
    pipe = joblib.load(path)
    has_per_mod = pipe.per_modality_transforms is not None
    has_tertiles = pipe.risk_tertiles is not None
    extras = []
    if has_per_mod:
        extras.append(f"per-modality x{len(pipe.per_modality_transforms)}")
    if has_tertiles:
        extras.append("tertiles")
    extras_str = f" [{', '.join(extras)}]" if extras else ""
    print(
        f"  [Pipeline] Loaded {pipe.model_name}/{pipe.imputation} "
        f"for {pipe.cohort.upper()} ({pipe.n_components} PCA components)"
        f"{extras_str}"
    )
    return pipe
