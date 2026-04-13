"""
Unified baseline runner for multimodal lung cancer survival prediction.
Strategy: selectable imputation + PCA-50 + selectable survival model.

Usage:
    python -m src.baseline.main_baseline    # defaults to CoxPH + zero imputation
    python -m src.baseline.main_baseline --model coxph --imputation zero
    python -m src.baseline.main_baseline --model coxph --imputation knn
    python -m src.baseline.main_baseline --model coxph --imputation knn_tuned
    python -m src.baseline.main_baseline --model coxph --imputation mice
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.baseline.models import CoxPHBaseline
from src.baseline.preprocessing import (
    IMPUTATION_STRATEGIES,
    apply_imputation,
    build_feature_matrix,
)
from src.data_loader import (
    MODALITY_KEYS,
    load_raw_data,
    load_split,
    load_split_patients,
)

DATA_DIR = Path("data/extracted/cache_data")
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = Path("results")

MODEL_CHOICES = ["coxph", "coxnet", "rsf", "xgboost"]


def _build_model(model_name: str):
    """Instantiate the selected survival model."""
    if model_name == "coxph":
        return CoxPHBaseline()
    elif model_name == "coxnet":
        from src.baseline.models import CoxNetModel
        return CoxNetModel()
    elif model_name == "rsf":
        # from src.baseline.models import RandomSurvivalForestModel
        # return RandomSurvivalForestModel()
        raise NotImplementedError("RSF not yet implemented.")
    elif model_name == "xgboost":
        from src.baseline.models import XGBoostSurvivalModel
        return XGBoostSurvivalModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _run_tag(model_name: str, imputation: str) -> str:
    """Build a filename-safe tag from model + imputation combination."""
    return f"{model_name}_{imputation}"


def c_index_by_completeness(
    is_complete: np.ndarray, X_pca: np.ndarray, y: np.ndarray, model
) -> dict:
    complete_idx = np.nonzero(is_complete)[0]
    incomplete_idx = np.nonzero(~is_complete)[0]
    results = {}

    for label, idx in [("complete", complete_idx), ("incomplete", incomplete_idx)]:
        if len(idx) < 5:
            print(f"  C-index ({label:10s}): N/A (n={len(idx)} < 5)")
            results[label] = None
            continue

        ci = model.score(X_pca[idx], y[idx])
        print(f"  C-index ({label:10s}, n={len(idx):3d}): {ci:.4f}")
        results[label] = ci

    return results


def save_results(results: list[dict], path: Path) -> None:
    """Save baseline results to JSON for later comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] JSON Results saved to {path}")


# ---------------------------------------------------------------------------
# Plot functions — categorized by dependency
# ---------------------------------------------------------------------------


def _plot_missingness(cohort: str, all_patients: list[dict]) -> None:
    """Data-dependent: identical across all models and imputations."""
    total_p = len(all_patients)
    missing_rates = {}
    for m in MODALITY_KEYS:
        missing_count = sum(1 for p in all_patients if p.get(m) is None)
        missing_rates[m] = (missing_count / total_p) * 100

    plt.figure(figsize=(8, 5))
    plt.bar(
        missing_rates.keys(), missing_rates.values(), color="coral", edgecolor="black"
    )
    plt.ylabel("Missing Data (%)")
    plt.title(f"Missing Modalities - TCGA-{cohort.upper()}")
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"plot_missingness_{cohort}.png")
    plt.close()


def _plot_kaplan_meier_completeness(
    cohort: str,
    y_test: np.ndarray,
    is_complete_test: np.ndarray,
) -> None:
    """
    Data-dependent: plots actual survival curves by modality completeness.
    Supports the MMNAR argument — shows that incomplete patients have
    different survival distributions than complete patients.
    """
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    idx_complete = np.nonzero(is_complete_test)[0]
    idx_incomplete = np.nonzero(~is_complete_test)[0]

    t_test = y_test["Time"]
    e_test = y_test["Status"]

    if len(idx_complete) > 0:
        kmf.fit(
            t_test[idx_complete],
            event_observed=e_test[idx_complete],
            label="Complete Data",
        )
        kmf.plot_survival_function(ci_show=True)

    if len(idx_incomplete) > 0:
        kmf.fit(
            t_test[idx_incomplete],
            event_observed=e_test[idx_incomplete],
            label="Incomplete Data",
        )
        kmf.plot_survival_function(ci_show=True)

    plt.title(f"Kaplan-Meier Survival Estimate (Test Set) - TCGA-{cohort.upper()}")
    plt.xlabel("Timeline (Days)")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"plot_kaplan_meier_{cohort}.png")
    plt.close()


def _plot_pca_variance(
    cohort: str,
    tag: str,
    pca: PCA,
    n_components: int,
) -> None:
    """
    Imputation-dependent: PCA is fitted after imputation, so the variance
    profile differs by strategy (19K dims for zero/knn vs ~200 for MICE).
    """
    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)

    plt.figure(figsize=(8, 6))
    plt.bar(
        range(1, n_components + 1),
        var_exp,
        alpha=0.5,
        align="center",
        label="Individual explained variance",
    )
    plt.step(
        range(1, n_components + 1),
        cum_var_exp,
        where="mid",
        label="Cumulative explained variance",
    )
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal component index")
    plt.title(f"PCA Variance Explained - TCGA-{cohort.upper()} [{tag.upper()}]")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"plot_pca_variance_{cohort}_{tag}.png")
    plt.close()


def _plot_kaplan_meier_risk(
    cohort: str,
    tag: str,
    y_test: np.ndarray,
    risk_scores: np.ndarray,
) -> None:
    """
    Model+imputation dependent: stratifies patients into High/Low risk
    groups based on the model's predicted risk scores (median split).
    Well-separated curves indicate the model discriminates effectively.
    """
    median_risk = np.median(risk_scores)
    high_risk = risk_scores >= median_risk
    low_risk = ~high_risk

    t_test = y_test["Time"]
    e_test = y_test["Status"]

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    if high_risk.sum() > 0:
        kmf.fit(
            t_test[high_risk],
            event_observed=e_test[high_risk],
            label="High Risk (predicted)",
        )
        kmf.plot_survival_function(ci_show=True)

    if low_risk.sum() > 0:
        kmf.fit(
            t_test[low_risk],
            event_observed=e_test[low_risk],
            label="Low Risk (predicted)",
        )
        kmf.plot_survival_function(ci_show=True)

    plt.title(f"Risk-Stratified KM (Test Set) - TCGA-{cohort.upper()} [{tag.upper()}]")
    plt.xlabel("Timeline (Days)")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"plot_kaplan_meier_risk_{cohort}_{tag}.png")
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_baseline(
    cohort: str, split_file: str, model_name: str, imputation: str
) -> dict:
    tag = _run_tag(model_name, imputation)

    print(f"\n{'=' * 60}")
    print(
        f"  Cohort: TCGA-{cohort.upper()}  |  "
        f"Model: {model_name.upper()}  |  Imputation: {imputation.upper()}"
    )
    print(f"{'=' * 60}")

    raw_data, _ = load_raw_data(DATA_DIR, cohort)
    train_ids, val_ids, test_ids = load_split(SPLITS_DIR, split_file)

    train_patients = load_split_patients(train_ids, raw_data)
    val_patients = load_split_patients(val_ids, raw_data)
    test_patients = load_split_patients(test_ids, raw_data)

    all_patients = train_patients + val_patients + test_patients

    # --- Build NaN feature matrices ---
    X_train, y_train, _ = build_feature_matrix(train_patients)
    X_val, y_val, is_complete_val = build_feature_matrix(val_patients)
    X_test, y_test, is_complete_test = build_feature_matrix(test_patients)

    print(f"  Train: {len(X_train)} patients | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Feature dim: {X_train.shape[1]}")
    print(
        f"  Event rate  — Train: {y_train['Status'].mean():.2%} | "
        f"Val: {y_val['Status'].mean():.2%} | Test: {y_test['Status'].mean():.2%}"
    )

    # --- Imputation ---
    print(f"\n  [Imputation: {imputation.upper()}]")
    (X_train_imp, X_val_imp, X_test_imp), imp_extra = apply_imputation(
        strategy=imputation,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
    )

    # --- PCA Dimensionality Reduction ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    n_components = min(50, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(
        f"  PCA components: {n_components} "
        f"(explained variance: {pca.explained_variance_ratio_.sum():.2%})"
    )

    # --- Model Selection & Training ---
    RESULTS_DIR.mkdir(exist_ok=True)
    model = _build_model(model_name)
    model.fit(X_train_pca, y_train)

    # --- Plots ---
    _plot_missingness(cohort, all_patients)  # data-dep
    _plot_kaplan_meier_completeness(cohort, y_test, is_complete_test)  # data-dep
    _plot_pca_variance(cohort, tag, pca, n_components)  # imp-dep

    risk_scores_test = model.predict_risk(X_test_pca)
    _plot_kaplan_meier_risk(cohort, tag, y_test, risk_scores_test)  # model+imp

    # --- Evaluation ---
    print("\n  [Overall Performance]")
    ci_train = model.score(X_train_pca, y_train)
    ci_val = model.score(X_val_pca, y_val)
    ci_test = model.score(X_test_pca, y_test)

    print(f"  C-index (train     ): {ci_train:.4f}")
    print(f"  C-index (val       ): {ci_val:.4f}")
    print(f"  C-index (test      ): {ci_test:.4f}")

    print("\n  [Breakdown by Modality Completeness - Val Set]")
    completeness_val = c_index_by_completeness(is_complete_val, X_val_pca, y_val, model)

    print("\n  [Breakdown by Modality Completeness - Test Set]")
    completeness_test = c_index_by_completeness(
        is_complete_test, X_test_pca, y_test, model
    )

    print("\n  [INFO] Plots saved to results/ directory.")

    result = {
        "cohort": cohort.upper(),
        "model": model_name,
        "imputation": imputation,
        "ci_train": ci_train,
        "ci_val": ci_val,
        "ci_test": ci_test,
        "ci_val_complete": completeness_val.get("complete"),
        "ci_val_incomplete": completeness_val.get("incomplete"),
        "ci_test_complete": completeness_test.get("complete"),
        "ci_test_incomplete": completeness_test.get("incomplete"),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "pca_n_components": n_components,
        "pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
    }

    if imp_extra:
        result["imputation_params"] = imp_extra

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline survival models with selectable imputation."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="coxph",
        choices=MODEL_CHOICES,
        help="Survival model to train (default: coxph).",
    )
    parser.add_argument(
        "--imputation",
        type=str,
        default="zero",
        choices=IMPUTATION_STRATEGIES,
        help="Imputation strategy for missing modalities (default: zero).",
    )
    args = parser.parse_args()

    tag = _run_tag(args.model, args.imputation)
    results = []

    results.append(
        run_baseline(
            cohort="luad",
            split_file="tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json",
            model_name=args.model,
            imputation=args.imputation,
        )
    )
    results.append(
        run_baseline(
            cohort="lusc",
            split_file="tcga_lusc_DSS_k5_r1_test0.2_val0.2_seed42.json",
            model_name=args.model,
            imputation=args.imputation,
        )
    )

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {args.model.upper()} + {args.imputation.upper()} + PCA-50")
    print(f"{'=' * 60}")

    for r in results:
        ci_comp = (
            f"{r['ci_test_complete']:.4f}"
            if r["ci_test_complete"] is not None
            else "N/A"
        )
        ci_incomp = (
            f"{r['ci_test_incomplete']:.4f}"
            if r["ci_test_incomplete"] is not None
            else "N/A"
        )
        print(
            f"  {r['cohort']}: C-index={r['ci_test']:.4f} "
            f"(Comp: {ci_comp} / Incomp: {ci_incomp})"
        )

        if r.get("imputation_params"):
            print(f"    Imputation params: {r['imputation_params']}")

    print(f"{'=' * 60}")
    save_results(results, RESULTS_DIR / f"baseline_results_{tag}.json")
