import warnings
from itertools import product

import numpy as np
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

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
    """
    Penalized Cox model (Elastic Net) via scikit-survival.
    l1_ratio=0.5 corresponds to an equal mix of L1 and L2 penalties.
    Alpha is selected via 5-fold CV on C-index over the regularization path.
    """
    def __init__(self, l1_ratio=0.5, alpha_min_ratio=0.1):
        self.l1_ratio = l1_ratio
        self.alpha_min_ratio = alpha_min_ratio
        self.model = None
        self._best_alpha_idx = None

    def fit(self, X, y):
        self.model = CoxnetSurvivalAnalysis(
            l1_ratio=self.l1_ratio,
            alpha_min_ratio=self.alpha_min_ratio,
            fit_baseline_model=True,
            normalize=False,  # It already comes normalized from the pipeline (StandardScaler + PCA)
        )
        self.model.fit(X, y)
        # We take the alpha in the middle of the path as the baseline reference point
        n_alphas = len(self.model.alphas_)
        self._best_alpha_idx = n_alphas // 2
        return self

    def predict_risk(self, X):
        preds = self.model.predict(X)
        if preds.ndim == 2:
            return preds[:, self._best_alpha_idx]
        return preds  # if it is 1D works directly

    def score(self, X, y):
        risk = self.predict_risk(X)  # already use _best_alpha_idx
        from sksurv.metrics import concordance_index_censored
        return concordance_index_censored(y["Status"], y["Time"], risk)[0]

class RandomSurvivalForestModel:
    pass


class XGBoostSurvivalModel:
    pass
