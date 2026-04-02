"""
Unified baseline runner for multimodal lung cancer survival prediction.
Strategy: zero-imputation + PCA-50 + selectable survival model.

Usage:
    python -m src.baseline.main_baseline                  # defaults to coxph
    python -m src.baseline.main_baseline --model coxph
    python -m src.baseline.main_baseline --model coxnet
    python -m src.baseline.main_baseline --model rsf
    python -m src.baseline.main_baseline --model xgboost
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
from src.baseline.preprocessing import build_structured_dataset
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
        # from src.baseline.models import CoxNetModel
        # return CoxNetModel()
        raise NotImplementedError("CoxNet not yet implemented.")
    elif model_name == "rsf":
        # from src.baseline.models import RandomSurvivalForestModel
        # return RandomSurvivalForestModel()
        raise NotImplementedError("RSF not yet implemented.")
    elif model_name == "xgboost":
        # from src.baseline.models import XGBoostSurvivalModel
        # return XGBoostSurvivalModel()
        raise NotImplementedError("XGBoost not yet implemented.")
    else:
        raise ValueError(f"Unknown model: {model_name}")


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


def _plot_data_diagnostics(
    cohort: str,
    all_patients: list[dict],
    pca: PCA,
    n_components: int,
) -> None:
    """
    Generate data-dependent plots (missingness + PCA variance).
    These are model-independent and identical across all runs.
    """
    # --- Missingness Bar Chart ---
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

    # --- Cumulative Variance Ratio (PCA) ---
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
    plt.title(f"PCA Variance Explained - TCGA-{cohort.upper()}")
    plt.ylim([0.0, 1.05])
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"plot_pca_variance_{cohort}.png")
    plt.close()


def _plot_kaplan_meier(
    cohort: str,
    model_name: str,
    y_test: np.ndarray,
    is_complete_test: np.ndarray,
) -> None:
    """Generate Kaplan-Meier survival curves for the test set (model-tagged)."""
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

    plt.title(
        f"Kaplan-Meier Survival Estimate (Test Set) - "
        f"TCGA-{cohort.upper()} [{model_name.upper()}]"
    )
    plt.xlabel("Timeline (Days)")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"plot_kaplan_meier_{cohort}_{model_name}.png")
    plt.close()


def run_baseline(cohort: str, split_file: str, model_name: str) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  Cohort: TCGA-{cohort.upper()}  |  Model: {model_name.upper()}")
    print(f"{'=' * 60}")

    raw_data, _ = load_raw_data(DATA_DIR, cohort)
    train_ids, val_ids, test_ids = load_split(SPLITS_DIR, split_file)

    train_patients = load_split_patients(train_ids, raw_data)
    val_patients = load_split_patients(val_ids, raw_data)
    test_patients = load_split_patients(test_ids, raw_data)

    all_patients = train_patients + val_patients + test_patients

    # --- Build Datasets (Zero Imputation) ---
    X_train, y_train, _ = build_structured_dataset(train_patients)
    X_val, y_val, is_complete_val = build_structured_dataset(val_patients)
    X_test, y_test, is_complete_test = build_structured_dataset(test_patients)

    event_rate_train = y_train["Status"].mean()
    event_rate_val = y_val["Status"].mean()
    event_rate_test = y_test["Status"].mean()

    print(f"  Train: {len(X_train)} patients | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Feature dim: {X_train.shape[1]}")
    print(
        f"  Event rate  — Train: {event_rate_train:.2%} | "
        f"Val: {event_rate_val:.2%} | Test: {event_rate_test:.2%}"
    )

    # --- PCA Dimensionality Reduction ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    n_components = min(50, X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(
        f"  PCA components: {n_components} "
        f"(explained variance: {pca.explained_variance_ratio_.sum():.2%})"
    )

    # --- Data-dependent plots (model-independent) ---
    RESULTS_DIR.mkdir(exist_ok=True)
    _plot_data_diagnostics(cohort, all_patients, pca, n_components)

    # --- Model Selection & Training ---
    model = _build_model(model_name)
    model.fit(X_train_pca, y_train)

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

    # --- Model-dependent plot ---
    _plot_kaplan_meier(cohort, model_name, y_test, is_complete_test)

    print("\n  [INFO] Plots saved to results/ directory.")

    return {
        "cohort": cohort.upper(),
        "model": model_name,
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
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline survival models (zero-imputation + PCA-50)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="coxph",
        choices=MODEL_CHOICES,
        help="Survival model to train (default: coxph).",
    )
    args = parser.parse_args()

    results = []

    results.append(
        run_baseline(
            cohort="luad",
            split_file="tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json",
            model_name=args.model,
        )
    )
    results.append(
        run_baseline(
            cohort="lusc",
            split_file="tcga_lusc_DSS_k5_r1_test0.2_val0.2_seed42.json",
            model_name=args.model,
        )
    )

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — {args.model.upper()} baseline (zero-imputation + PCA-50)")
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

    print(f"{'=' * 60}")
    save_results(results, RESULTS_DIR / f"baseline_results_{args.model}.json")
