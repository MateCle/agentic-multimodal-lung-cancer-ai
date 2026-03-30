"""
Naive ML baseline for multimodal lung cancer survival prediction.
Strategy: zero-imputation for missing modalities + early fusion + CoxPH.
Outputs C-index, AUC, and generates diagnostic plots (Missingness, PCA, Kaplan-Meier, ROC curve).
Includes breakdown by data completeness and JSON result logging.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader import (
    MODALITY_DIMS,
    MODALITY_KEYS,
    load_raw_data,
    load_split,
    load_split_patients,
)

DATA_DIR = Path("data/extracted/cache_data")
SPLITS_DIR = DATA_DIR / "splits"

SURVIVAL_EVENT = "DSS"
SURVIVAL_TIME = "DSS.time"


def zero_impute(patient: dict) -> np.ndarray:
    """
    Concatenate all modality features using zero-imputation for missing ones.
    Strictly enforces dimension checks to prevent inhomogeneous arrays.
    Returns a single flat feature vector.
    """
    vectors = []

    for modality in MODALITY_KEYS:
        expected_dim = MODALITY_DIMS[modality]
        val = patient.get(modality)

        if val is not None:
            val_arr = np.array(val).flatten()
            if val_arr.size == expected_dim:
                vectors.append(val_arr)
            else:
                vectors.append(np.zeros(expected_dim, dtype=np.float32))
        else:
            vectors.append(np.zeros(expected_dim, dtype=np.float32))

    return np.concatenate(vectors)


def build_dataset(
    patients: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix X, survival labels (events, times), and completeness mask.
    Filters out patients with missing or invalid survival labels.
    """
    X, events, times, is_complete = [], [], [], []

    for p in patients:
        label = p["label"]
        event = label.get(SURVIVAL_EVENT)
        time = label.get(SURVIVAL_TIME)

        if event is None or time is None:
            continue
        if np.isnan(event) or np.isnan(time):
            continue
        if time <= 0:
            continue

        X.append(zero_impute(p))
        events.append(int(event))
        times.append(float(time))

        missing_modalities = [m for m in MODALITY_KEYS if p.get(m) is None]
        is_complete.append(len(missing_modalities) == 0)

    return np.array(X), np.array(events), np.array(times), np.array(is_complete)


def c_index_by_completeness(
    is_complete: np.ndarray,
    X_pca: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    cph: CoxPHFitter,
    cols: list,
) -> dict:
    """
    Compute C-index separately for patients with complete vs incomplete data.
    """
    complete_idx = np.nonzero(is_complete)[0]
    incomplete_idx = np.nonzero(~is_complete)[0]

    results = {}

    for label, idx in [("complete", complete_idx), ("incomplete", incomplete_idx)]:
        if len(idx) < 5:
            print(f"  C-index ({label:10s}): N/A (n={len(idx)} < 5)")
            results[label] = None
            continue

        df = pd.DataFrame(X_pca[idx], columns=cols)
        risk_scores = cph.predict_partial_hazard(df).values
        ci = concordance_index(times[idx], -risk_scores, events[idx])

        print(f"  C-index ({label:10s}, n={len(idx):3d}): {ci:.4f}")
        results[label] = ci

    return results


def save_results(results: list[dict], path: Path) -> None:
    """Save baseline results to JSON for later comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] JSON Results saved to {path}")


def run_baseline(cohort: str, split_file: str) -> dict:
    """
    Train and evaluate the naive baseline for a single cohort.
    Generates diagnostic plots and returns model performance metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"  Cohort: TCGA-{cohort.upper()}")
    print(f"{'=' * 60}")

    raw_data, _ = load_raw_data(DATA_DIR, cohort)
    train_ids, val_ids, test_ids = load_split(SPLITS_DIR, split_file)

    train_patients = load_split_patients(train_ids, raw_data)
    val_patients = load_split_patients(val_ids, raw_data)
    test_patients = load_split_patients(test_ids, raw_data)

    # --- PLOT 1: Missingness Bar Chart ---
    all_patients = train_patients + val_patients + test_patients
    missing_rates = {}
    total_p = len(all_patients)

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
    plt.savefig(Path(f"results/plot_missingness_{cohort}.png"))
    plt.close()

    # --- Build Datasets ---
    X_train, e_train, t_train, _ = build_dataset(train_patients)
    X_val, e_val, t_val, is_complete_val = build_dataset(val_patients)
    X_test, e_test, t_test, is_complete_test = build_dataset(test_patients)

    print(f"  Train: {len(X_train)} patients | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Feature dim: {X_train.shape[1]}")
    print(
        f"  Event rate  — Train: {e_train.mean():.2%} | "
        f"Val: {e_val.mean():.2%} | Test: {e_test.mean():.2%}"
    )

    # --- PCA Dimensionality Reduction ---
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

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

    # --- PLOT 2: Cumulative Variance Ratio (PCA) ---
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
    plt.savefig(Path(f"results/plot_pca_variance_{cohort}.png"))
    plt.close()

    # --- Cox Proportional Hazards Model ---
    cols = [f"pc{i}" for i in range(n_components)]

    df_train = pd.DataFrame(X_train_pca, columns=cols)
    df_train[SURVIVAL_EVENT] = e_train
    df_train[SURVIVAL_TIME] = t_train

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_train, duration_col=SURVIVAL_TIME, event_col=SURVIVAL_EVENT)

    def evaluate_c_index(X_pca, events, times, split_name):
        df = pd.DataFrame(X_pca, columns=cols)
        risk_scores = cph.predict_partial_hazard(df).values
        ci = concordance_index(times, -risk_scores, events)
        print(f"  C-index ({split_name:10s}): {ci:.4f}")
        return ci

    print("\n  [Overall Performance]")
    ci_train = evaluate_c_index(X_train_pca, e_train, t_train, "train")
    ci_val = evaluate_c_index(X_val_pca, e_val, t_val, "val")
    ci_test = evaluate_c_index(X_test_pca, e_test, t_test, "test")

    # --- AUC Calculation ---
    def evaluate_auc(X_pca, events, split_name):
        df = pd.DataFrame(X_pca, columns=cols)
        risk_scores = cph.predict_partial_hazard(df).values
        if len(np.unique(events)) < 2:
            print(f"  AUC     ({split_name:10s}): N/A (single class)")
            return None
        auc = roc_auc_score(events, risk_scores)
        print(f"  AUC     ({split_name:10s}): {auc:.4f}")
        return auc

    auc_train = evaluate_auc(X_train_pca, e_train, "train")
    auc_val = evaluate_auc(X_val_pca, e_val, "val")
    auc_test = evaluate_auc(X_test_pca, e_test, "test")

    # --- Breakdown by Completeness ---
    print("\n  [Breakdown by Modality Completeness - Val Set]")
    completeness_val = c_index_by_completeness(
        is_complete_val, X_val_pca, e_val, t_val, cph, cols
    )

    print("\n  [Breakdown by Modality Completeness - Test Set]")
    completeness_test = c_index_by_completeness(
        is_complete_test, X_test_pca, e_test, t_test, cph, cols
    )

    # --- PLOT 3: Kaplan-Meier Survival Curves (Test Set) ---
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    idx_complete = np.nonzero(is_complete_test)[0]
    idx_incomplete = np.nonzero(~is_complete_test)[0]

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
    plt.savefig(Path(f"results/plot_kaplan_meier_{cohort}.png"))
    plt.close()

    # --- PLOT 4: ROC Curve (Test Set) ---
    if len(np.unique(e_test)) >= 2 and auc_test is not None:
        df_test_roc = pd.DataFrame(X_test_pca, columns=cols)
        risk_scores_test = cph.predict_partial_hazard(df_test_roc).values

        fpr, tpr, _ = roc_curve(e_test, risk_scores_test)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {auc_test:.4f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Test Set) - TCGA-{cohort.upper()}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(Path(f"results/plot_roc_curve_{cohort}.png"))
        plt.close()

    print("\n  [INFO] Plots saved to results/ directory.")

    return {
        "cohort": cohort.upper(),
        "ci_train": ci_train,
        "ci_val": ci_val,
        "ci_test": ci_test,
        "auc_train": auc_train,
        "auc_val": auc_val,
        "auc_test": auc_test,
        "ci_val_complete": completeness_val.get("complete"),
        "ci_val_incomplete": completeness_val.get("incomplete"),
        "ci_test_complete": completeness_test.get("complete"),
        "ci_test_incomplete": completeness_test.get("incomplete"),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }


if __name__ == "__main__":
    results = []

    results.append(
        run_baseline(
            cohort="luad",
            split_file="tcga_luad_DSS_k3_r1_test0.2_val0.2_seed42.json",
        )
    )

    results.append(
        run_baseline(
            cohort="lusc",
            split_file="tcga_lusc_DSS_k5_r1_test0.2_val0.2_seed42.json",
        )
    )

    print(f"\n{'=' * 60}")
    print("  SUMMARY — Naive baseline (zero-imputation + CoxPH + PCA-50)")
    print(f"{'=' * 60}")

    for r in results:
        auc_str = f"{r['auc_test']:.4f}" if r["auc_test"] is not None else "N/A"
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
            f"  {r['cohort']}: "
            f"C-index={r['ci_test']:.4f} (Comp: {ci_comp} / Incomp: {ci_incomp}) | "
            f"AUC={auc_str}"
        )

    print(f"{'=' * 60}")

    save_results(results, Path("results/baseline_results.json"))
