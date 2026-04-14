"""
SHAP explainability module for baseline survival models.
Computes SHAP values on PCA components, then back-projects through
PCA loadings to identify the most important original features.

Back-projection formula:
    feature_shap[j] = mean(|sum_k(shap_pca[i,k] * pca.components_[k,j])|)
    over all samples i.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.decomposition import PCA

from src.data_loader import MODALITY_DIMS, MODALITY_KEYS


def _build_feature_names(n_features: int, metadata: dict = None) -> list[str]:
    """
    Generate human-readable feature names based on modality blocks.

    If metadata is provided (from tcga_*_metadata.pkl), uses real column
    names: clinical variable names, REACTOME pathway names, CpG probe IDs.
    WSI features have no column names in metadata, so they get 'wsi_0', etc.

    If metadata is not provided, falls back to '{modality}_{index}' names.
    If n_features doesn't match the expected total, uses 'feat_{index}'.
    """
    expected_total = sum(MODALITY_DIMS.values())

    if n_features != expected_total:
        return [f"feat_{i}" for i in range(n_features)]

    # Map from modality key to metadata column key
    _META_KEYS = {
        "clinical": "clinical_columns",
        "transcriptomics": "transcriptomics_columns",
        "methylation": "methylation_columns",
    }

    names = []
    for mod in MODALITY_KEYS:
        dim = MODALITY_DIMS[mod]
        meta_key = _META_KEYS.get(mod)

        if metadata and meta_key and meta_key in metadata:
            col_names = metadata[meta_key]
            if len(col_names) == dim:
                names.extend(col_names)
            else:
                names.extend([f"{mod}_{i}" for i in range(dim)])
        else:
            # WSI or missing metadata: positional names
            names.extend([f"{mod}_{i}" for i in range(dim)])

    return names


def _get_shap_values(model, X_pca: np.ndarray, model_name: str) -> np.ndarray:
    """
    Compute SHAP values on PCA components.
    Uses TreeExplainer for tree models (fast, exact) and
    a sampling-based Explainer for linear models.

    Returns:
        shap_values: (n_samples, n_components) array.
    """
    if model_name in ("xgboost", "rsf"):
        explainer = shap.TreeExplainer(model.model)
        sv = explainer.shap_values(X_pca)
    else:
        # CoxPH / CoxNet: use predict_risk as the model function
        # with a small background sample for efficiency
        bg = shap.sample(X_pca, min(50, len(X_pca)))
        explainer = shap.Explainer(model.predict_risk, bg)
        sv = explainer(X_pca).values

    return np.array(sv)


def compute_shap_importance(
    model,
    model_name: str,
    X_test_pca: np.ndarray,
    pca: PCA,
    n_top: int = 20,
    metadata: dict = None,
) -> dict:
    """
    Compute SHAP importance at both PCA component and original feature level.

    Args:
        model:       Fitted model with .predict_risk() method.
        model_name:  String identifier ('coxph', 'xgboost', etc.).
        X_test_pca:  Test set in PCA space, shape (n_samples, n_components).
        pca:         Fitted PCA object with .components_ attribute.
        n_top:       Number of top features to return.
        metadata:    Optional dict from tcga_*_metadata.pkl with column names.

    Returns:
        Dict with:
            'component_importance': mean |SHAP| per PCA component.
            'feature_importance':   mean |SHAP| per original feature (back-projected).
            'top_features':         List of (feature_name, importance) tuples.
            'shap_values_pca':      Raw SHAP values on PCA components.
    """
    print("  [SHAP] Computing SHAP values on PCA components...")
    shap_values = _get_shap_values(model, X_test_pca, model_name)

    # Mean absolute SHAP per PCA component
    component_importance = np.mean(np.abs(shap_values), axis=0)

    # Back-project to original feature space
    # shap_values: (n_samples, n_components)
    # pca.components_: (n_components, n_original_features)
    print("  [SHAP] Back-projecting through PCA loadings...")
    back_projected = shap_values @ pca.components_
    feature_importance = np.mean(np.abs(back_projected), axis=0)

    # Build feature names and rank
    n_original = pca.components_.shape[1]
    feature_names = _build_feature_names(n_original, metadata=metadata)

    top_idx = np.argsort(feature_importance)[::-1][:n_top]
    top_features = [(feature_names[i], float(feature_importance[i])) for i in top_idx]

    print(f"  [SHAP] Top {n_top} features by back-projected importance:")
    for rank, (name, imp) in enumerate(top_features, 1):
        print(f"    {rank:2d}. {name:30s}  {imp:.6f}")

    return {
        "component_importance": component_importance,
        "feature_importance": feature_importance,
        "top_features": top_features,
        "shap_values_pca": shap_values,
    }


def plot_shap(
    shap_result: dict,
    cohort: str,
    tag: str,
    results_dir: Path,
    n_top: int = 20,
) -> None:
    """
    Generate SHAP visualizations:
    1. PCA component importance (bar chart).
    2. Top original features after back-projection (horizontal bar chart).
    """
    # --- Plot 1: PCA Component Importance ---
    comp_imp = shap_result["component_importance"]
    n_comp = len(comp_imp)

    plt.figure(figsize=(10, 5))
    plt.bar(range(n_comp), comp_imp, color="steelblue", edgecolor="black")
    plt.xlabel("PCA Component")
    plt.ylabel("Mean |SHAP value|")
    plt.title(f"SHAP Component Importance - TCGA-{cohort.upper()} [{tag.upper()}]")
    plt.tight_layout()
    plt.savefig(results_dir / f"plot_shap_components_{cohort}_{tag}.png")
    plt.close()

    # --- Plot 2: Top Original Features (Back-Projected) ---
    top_features = shap_result["top_features"][:n_top]
    names = [name for name, _ in reversed(top_features)]
    values = [imp for _, imp in reversed(top_features)]

    plt.figure(figsize=(10, 8))
    plt.barh(names, values, color="coral", edgecolor="black")
    plt.xlabel("Mean |SHAP value| (back-projected)")
    plt.title(
        f"Top {n_top} Features by SHAP Importance - "
        f"TCGA-{cohort.upper()} [{tag.upper()}]"
    )
    plt.tight_layout()
    plt.savefig(results_dir / f"plot_shap_features_{cohort}_{tag}.png")
    plt.close()
