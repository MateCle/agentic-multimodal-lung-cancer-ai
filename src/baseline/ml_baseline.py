"""
Naive ML baseline for multimodal lung cancer survival prediction.
Strategy: zero-imputation for missing modalities + early fusion + CoxPH.
Outputs C-index and AUCon the test split as the benchmark for the agentic system.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

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


def build_dataset(patients: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix X and survival labels (event, time) from patient list.
    Filters out patients with missing survival labels.
    """
    X, events, times = [], [], []

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

    return np.array(X), np.array(events), np.array(times)


def run_baseline(cohort: str, split_file: str) -> dict:
    """
    Train and evaluate the naive baseline for a single cohort.
    Returns a dict with train/val/test C-index and AUC scores.
    """
    print(f"\n{'=' * 60}")
    print(f"  Cohort: TCGA-{cohort.upper()}")
    print(f"{'=' * 60}")

    raw_data, _ = load_raw_data(DATA_DIR, cohort)
    train_ids, val_ids, test_ids = load_split(SPLITS_DIR, split_file)

    train_patients = load_split_patients(train_ids, raw_data)
    val_patients = load_split_patients(val_ids, raw_data)
    test_patients = load_split_patients(test_ids, raw_data)

    X_train, e_train, t_train = build_dataset(train_patients)
    X_val, e_val, t_val = build_dataset(val_patients)
    X_test, e_test, t_test = build_dataset(test_patients)

    print(f"  Train: {len(X_train)} patients | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Feature dim: {X_train.shape[1]}")
    print(
        f"  Event rate  — Train: {e_train.mean():.2%} | "
        f"Val: {e_val.mean():.2%} | Test: {e_test.mean():.2%}"
    )

    # --- CoxPH on PCA-reduced features ---
    # CoxPH cannot handle 19077-dim features directly.
    # We reduce with PCA to 50 components before fitting.
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

    # build DataFrame for lifelines
    cols = [f"pc{i}" for i in range(n_components)]

    df_train = pd.DataFrame(X_train_pca, columns=cols)
    df_train[SURVIVAL_EVENT] = e_train
    df_train[SURVIVAL_TIME] = t_train

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_train, duration_col=SURVIVAL_TIME, event_col=SURVIVAL_EVENT)

    def c_index_on(X_pca, events, times, label):
        df = pd.DataFrame(X_pca, columns=cols)
        risk_scores = cph.predict_partial_hazard(df).values
        ci = concordance_index(times, -risk_scores, events)
        print(f"  C-index ({label:5s}): {ci:.4f}")
        return ci

    ci_train = c_index_on(X_train_pca, e_train, t_train, "train")
    ci_val = c_index_on(X_val_pca, e_val, t_val, "val")
    ci_test = c_index_on(X_test_pca, e_test, t_test, "test")

    # --- AUC CALCULATION ---
    def auc_on(x_pca_local, events_local, label):
        """AUC treating DSS as binary classification."""
        df_local = pd.DataFrame(x_pca_local, columns=cols)
        risk_scores = cph.predict_partial_hazard(df_local).values
        if len(np.unique(events_local)) < 2:
            print(f"  AUC     ({label:5s}): N/A (single class)")
            return None
        auc = roc_auc_score(events_local, risk_scores)
        print(f"  AUC     ({label:5s}): {auc:.4f}")
        return auc

    auc_train = auc_on(X_train_pca, e_train, "train")
    auc_val = auc_on(X_val_pca, e_val, "val")
    auc_test = auc_on(X_test_pca, e_test, "test")

    return {
        "cohort": cohort.upper(),
        "ci_train": ci_train,
        "ci_val": ci_val,
        "ci_test": ci_test,
        "auc_train": auc_train,
        "auc_val": auc_val,
        "auc_test": auc_test,
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
        print(f"  {r['cohort']}: C-index test={r['ci_test']:.4f} | AUC test={auc_str}")
    print(f"{'=' * 60}")
