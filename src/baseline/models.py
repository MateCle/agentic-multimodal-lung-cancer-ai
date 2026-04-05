import warnings
from itertools import product

import numpy as np
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis


class CoxPHBaseline:
    """
    Naive Cox Proportional Hazards model using scikit-survival.
    Includes a small L2 penalty (alpha) to match lifelines' default penalizer behavior.
    """

    def __init__(self, alpha=0.1):
        self.model = CoxPHSurvivalAnalysis(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_risk(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        """Returns the Concordance Index (C-index)"""
        return self.model.score(X, y)


class CoxNetModel:
    pass


class RandomSurvivalForestModel:
    """
    Random Survival Forest wrapper with optional hyperparameter tuning
    via 5-fold cross-validation on the training set.

    Default behaviour (tuned=False): fits a single RSF with sensible defaults.
    Tuned behaviour  (tuned=True):  grid-searches over n_estimators,
    max_depth, min_samples_split, min_samples_leaf and max_features,
    selecting the combination that maximises mean CV C-index.

    The wrapper exposes the same .fit / .predict_risk / .score interface
    used by CoxPHBaseline, so it plugs straight into main_baseline.py.
    """

    # ------------------------------------------------------------------
    # Hyperparameter grid (kept compact to run in reasonable time)
    # ------------------------------------------------------------------
    PARAM_GRID = {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, None],
        "min_samples_split": [6, 10, 20],
        "min_samples_leaf": [3, 6, 10],
        "max_features": ["sqrt", "log2"],
    }

    def __init__(self, tuned: bool = False, random_state: int = 42, n_jobs: int = -1):
        self.tuned = tuned
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_params: dict = {}
        self.cv_results: list[dict] = []
        self.model: RandomSurvivalForest | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Train the model.  If self.tuned is True, run a 5-fold CV grid
        search first and refit on the full training set with the best
        hyperparameters found.
        """
        if self.tuned:
            self._tune_and_fit(X, y)
        else:
            self.model = RandomSurvivalForest(
                n_estimators=300,
                max_depth=10,
                min_samples_split=6,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            self.model.fit(X, y)
        return self

    def predict_risk(self, X):
        """Return per-sample risk scores (higher = worse prognosis)."""
        return self.model.predict(X)

    def score(self, X, y):
        """Return Harrell's concordance index."""
        return self.model.score(X, y)

    # ------------------------------------------------------------------
    # Hyperparameter tuning (private)
    # ------------------------------------------------------------------

    def _tune_and_fit(self, X, y):
        """
        5-fold CV grid search over PARAM_GRID.
        Scoring: C-index via RSF.score() on the held-out fold.
        After tuning, refit the best model on the full training data.
        """
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Generate all combinations
        keys = list(self.PARAM_GRID.keys())
        values = list(self.PARAM_GRID.values())
        combos = list(product(*values))

        best_ci = -np.inf
        best_params = {}

        print(f"\n  [RSF Tuning — 5-Fold CV, {len(combos)} combos]")

        for combo in combos:
            params = dict(zip(keys, combo))
            fold_scores = []

            for tr_idx, va_idx in kf.split(X):
                try:
                    X_tr, X_va = X[tr_idx], X[va_idx]
                    y_tr, y_va = y[tr_idx], y[va_idx]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rsf = RandomSurvivalForest(
                            **params,
                            n_jobs=self.n_jobs,
                            random_state=self.random_state,
                        )
                        rsf.fit(X_tr, y_tr)
                        fold_scores.append(rsf.score(X_va, y_va))
                except Exception:
                    continue

            if not fold_scores:
                continue

            mean_ci = float(np.mean(fold_scores))
            std_ci = float(np.std(fold_scores))

            self.cv_results.append({**params, "mean_ci": mean_ci, "std_ci": std_ci})

            # Compact summary for each combo
            depth_str = str(params["max_depth"]) if params["max_depth"] else "∞"
            print(
                f"    n={params['n_estimators']:3d}  depth={depth_str:>3s}  "
                f"split={params['min_samples_split']:2d}  leaf={params['min_samples_leaf']:2d}  "
                f"feat={params['max_features']:<4s}  →  "
                f"CV C-index={mean_ci:.4f} ± {std_ci:.4f}"
            )

            if mean_ci > best_ci:
                best_ci = mean_ci
                best_params = params

        self.best_params = best_params
        print(f"\n  → Best RSF params: {best_params}")
        print(f"  → Best CV C-index: {best_ci:.4f}")

        # Refit on full training data with the best hyperparameters
        self.model = RandomSurvivalForest(
            **best_params,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.model.fit(X, y)


class XGBoostSurvivalModel:
    pass
