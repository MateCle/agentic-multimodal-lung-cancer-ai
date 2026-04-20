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
) -> Path:
    """
    Serialize a fitted pipeline to disk.

    Args:
        model:       Fitted survival model with .predict_risk() method.
        scaler:      Fitted StandardScaler.
        pca:         Fitted PCA transformer.
        cohort:      'luad' or 'lusc'.
        model_name:  'coxph', 'xgboost', etc.
        imputation:  'zero', 'knn', 'knn_tuned', 'mice'.

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
    print(
        f"  [Pipeline] Loaded {pipe.model_name}/{pipe.imputation} "
        f"for {pipe.cohort.upper()} ({pipe.n_components} PCA components)"
    )
    return pipe
