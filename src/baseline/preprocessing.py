"""
Preprocessing module for multimodal lung cancer survival prediction.
Handles feature extraction, survival label formatting, and imputation strategies.

Imputation strategies:
    - zero:      Replace missing modalities with zero vectors.
    - knn:       K-Nearest Neighbors imputation (fit on train only).
    - knn_tuned: KNN with hyperparameters selected via 5-fold CV on train set.
    - mice:      Multiple Imputation by Chained Equations (IterativeImputer),
                 with per-modality PCA reduction for scalability.
"""

import warnings
from itertools import product

import numpy as np
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis

from src.data_loader import MODALITY_DIMS, MODALITY_KEYS

SURVIVAL_EVENT = "DSS"
SURVIVAL_TIME = "DSS.time"

IMPUTATION_STRATEGIES = ["zero", "knn", "knn_tuned", "mice"]


# ---------------------------------------------------------------------------
# Feature extraction (strategy-agnostic)
# ---------------------------------------------------------------------------


def _concat_features(patient: dict) -> np.ndarray:
    """
    Concatenate all modality features into a single flat vector.
    Missing modalities are filled with NaN so that any imputation
    strategy can be applied downstream on the full matrix.
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
                vectors.append(np.full(expected_dim, np.nan, dtype=np.float32))
        else:
            vectors.append(np.full(expected_dim, np.nan, dtype=np.float32))

    return np.concatenate(vectors)


def build_feature_matrix(
    patients: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix X (with NaN for missing modalities),
    structured survival labels y, and a boolean completeness mask.

    Returns:
        X:           (n_patients, n_features) with NaN for missing modalities.
        y:           Structured array [('Status', bool), ('Time', float)]
                     compatible with scikit-survival.
        is_complete: Boolean array, True if all modalities are present.
    """
    X, y_list, is_complete = [], [], []

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

        X.append(_concat_features(p))
        y_list.append((bool(event), float(time)))

        missing = [m for m in MODALITY_KEYS if p.get(m) is None]
        is_complete.append(len(missing) == 0)

    y = np.array(y_list, dtype=[("Status", "?"), ("Time", "<f8")])
    return np.array(X), y, np.array(is_complete)


# ---------------------------------------------------------------------------
# Modality index ranges
# ---------------------------------------------------------------------------


def _modality_ranges() -> dict[str, tuple[int, int]]:
    """Return the column index range (start, end) for each modality."""
    ranges = {}
    offset = 0
    for mod in MODALITY_KEYS:
        dim = MODALITY_DIMS[mod]
        ranges[mod] = (offset, offset + dim)
        offset += dim
    return ranges


# ---------------------------------------------------------------------------
# Imputation strategies (private)
# ---------------------------------------------------------------------------


def _impute_zero(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace NaN with zeros."""
    return (
        np.nan_to_num(X_train),
        np.nan_to_num(X_val),
        np.nan_to_num(X_test),
    )


def _impute_knn(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_neighbors: int = 5,
    weights: str = "distance",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KNN imputation fitted on training set only (no data leakage)."""
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    return (
        imputer.fit_transform(X_train),
        imputer.transform(X_val),
        imputer.transform(X_test),
    )


def _impute_knn_tuned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Tune KNN imputer hyperparameters via 5-fold CV on the training set.
    Scoring: CoxPH C-index on the held-out fold (impute -> scale -> PCA -> CoxPH).
    Returns imputed matrices and the best parameter dict.

    Uses alpha=1.0 for CoxPH to suppress numerical overflow on small folds.
    """
    n_neighbors_grid = [3, 5, 7, 10, 15]
    weights_grid = ["uniform", "distance"]

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    best_ci = -np.inf
    best_params = {"n_neighbors": 5, "weights": "distance"}

    print("\n  [KNN Tuning — 5-Fold CV on Train Set]")

    for n_neighbors, weights in product(n_neighbors_grid, weights_grid):
        fold_scores = []

        for tr_idx, va_idx in kf.split(X_train):
            try:
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]

                imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
                X_tr_imp = imputer.fit_transform(X_tr)
                X_va_imp = imputer.transform(X_va)

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr_imp)
                X_va_s = scaler.transform(X_va_imp)

                n_comp = min(50, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
                pca = PCA(n_components=n_comp, random_state=42)
                X_tr_pca = pca.fit_transform(X_tr_s)
                X_va_pca = pca.transform(X_va_s)

                # Higher alpha suppresses overflow in exp() on small CV folds
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="overflow", category=RuntimeWarning
                    )
                    cph = CoxPHSurvivalAnalysis(alpha=1.0)
                    cph.fit(X_tr_pca, y_tr)
                    fold_scores.append(cph.score(X_va_pca, y_va))
            except Exception:
                continue

        if not fold_scores:
            continue

        mean_ci = np.mean(fold_scores)
        print(f"    k={n_neighbors:2d}, w={weights:8s} → CV C-index={mean_ci:.4f}")

        # Prefer higher C-index; break ties by smaller k (more stable)
        if mean_ci > best_ci + 1e-4 or (
            abs(mean_ci - best_ci) <= 1e-4 and n_neighbors < best_params["n_neighbors"]
        ):
            best_ci = mean_ci
            best_params = {"n_neighbors": n_neighbors, "weights": weights}

    print(
        f"  → Best: k={best_params['n_neighbors']}, "
        f"w={best_params['weights']} (CV C-index={best_ci:.4f})"
    )

    imputer = KNNImputer(**best_params)
    return (
        imputer.fit_transform(X_train),
        imputer.transform(X_val),
        imputer.transform(X_test),
        best_params,
    )


def _impute_mice(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components_per_modality: int = 50,
    max_iter: int = 10,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MICE imputation with per-modality PCA reduction for scalability.

    Standard MICE (IterativeImputer with BayesianRidge) cannot scale to
    19K features because it computes a full p×p covariance matrix.
    This implementation first reduces each modality independently via PCA,
    producing a ~200-dim matrix (50 per modality × 4 modalities), then
    runs MICE on the reduced space.

    The per-modality PCA is fitted only on training patients who have
    that modality, avoiding any information from missing values.
    """
    ranges = _modality_ranges()

    reduced_splits = {"train": [], "val": [], "test": []}
    splits = {"train": X_train, "val": X_val, "test": X_test}

    print(
        f"  [MICE] Per-modality PCA reduction ({n_components_per_modality} components each)"
    )

    for mod in MODALITY_KEYS:
        start, end = ranges[mod]

        # Extract this modality's columns from each split
        X_mod = {name: X[:, start:end] for name, X in splits.items()}

        # Identify training patients who have this modality (no NaN in this block)
        has_mod_train = ~np.isnan(X_mod["train"]).any(axis=1)
        n_available = has_mod_train.sum()

        if n_available < 2:
            # Not enough data to fit PCA — fill with NaN
            n_comp = min(n_components_per_modality, end - start)
            for name in splits:
                reduced_splits[name].append(
                    np.full((len(splits[name]), n_comp), np.nan)
                )
            print(f"    {mod}: skipped (only {n_available} patients available)")
            continue

        # Fit scaler + PCA on available training patients only
        scaler = StandardScaler()
        X_avail = X_mod["train"][has_mod_train]
        scaler.fit(X_avail)

        n_comp = min(n_components_per_modality, X_avail.shape[0] - 1, X_avail.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        pca.fit(scaler.transform(X_avail))

        explained = pca.explained_variance_ratio_.sum()
        print(f"    {mod}: {n_comp} components ({explained:.1%} variance explained)")

        # Transform each split: real values for patients who have the modality,
        # NaN for patients who are missing it
        for name, X_full in splits.items():
            X_m = X_full[:, start:end]
            has_mod = ~np.isnan(X_m).any(axis=1)
            reduced = np.full((len(X_full), n_comp), np.nan)
            if has_mod.any():
                reduced[has_mod] = pca.transform(scaler.transform(X_m[has_mod]))
            reduced_splits[name].append(reduced)

    # Concatenate per-modality PCA embeddings
    X_train_reduced = np.hstack(reduced_splits["train"])
    X_val_reduced = np.hstack(reduced_splits["val"])
    X_test_reduced = np.hstack(reduced_splits["test"])

    total_dim = X_train_reduced.shape[1]
    print(f"  [MICE] Reduced feature space: {total_dim} dimensions")
    print(f"  [MICE] Running IterativeImputer (max_iter={max_iter})...")

    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=False,
    )
    return (
        imputer.fit_transform(X_train_reduced),
        imputer.transform(X_val_reduced),
        imputer.transform(X_test_reduced),
    )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def apply_imputation(
    strategy: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    """
    Apply the selected imputation strategy.

    Args:
        strategy:  One of 'zero', 'knn', 'knn_tuned', 'mice'.
        X_train:   Training feature matrix (with NaN for missing modalities).
        X_val:     Validation feature matrix.
        X_test:    Test feature matrix.
        y_train:   Structured survival labels for train set.
                   Required only for 'knn_tuned' (used in CV scoring).

    Returns:
        (X_train_imp, X_val_imp, X_test_imp): Imputed feature matrices.
            NOTE: output dimensionality may differ from input when
            per-modality PCA is applied (e.g. MICE returns ~200 dims).
        extra: Dict with additional info (e.g. best_params for knn_tuned).
    """
    if strategy == "zero":
        return _impute_zero(X_train, X_val, X_test), {}

    elif strategy == "knn":
        return _impute_knn(X_train, X_val, X_test), {}

    elif strategy == "knn_tuned":
        if y_train is None:
            raise ValueError("y_train is required for knn_tuned strategy.")
        X_tr, X_va, X_te, params = _impute_knn_tuned(X_train, y_train, X_val, X_test)
        return (X_tr, X_va, X_te), params

    elif strategy == "mice":
        return _impute_mice(X_train, X_val, X_test), {}

    else:
        raise ValueError(
            f"Unknown imputation strategy: '{strategy}'. "
            f"Choose from {IMPUTATION_STRATEGIES}."
        )
