import numpy as np
import xgboost as xgb
from sksurv.linear_model import CoxPHSurvivalAnalysis
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
    pass


class RandomSurvivalForestModel:
    pass


class XGBoostSurvivalModel:
    """
    XGBoost survival model using the Cox proportional hazards objective.

    XGBoost's survival:cox objective expects labels encoded as:
        +time if event observed (uncensored)
        -time if censored
    Higher predicted values = higher risk.
    """

    def __init__(
        self,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
        }
        self.model = None

    @staticmethod
    def _encode_survival_labels(y: np.ndarray) -> np.ndarray:
        """
        Convert scikit-survival structured array to XGBoost survival format.
        Structured array: [('Status', bool), ('Time', float)]
        XGBoost format:   +time (event) / -time (censored)
        """
        times = y["Time"]
        events = y["Status"]
        return np.where(events, times, -times)

    def fit(self, X, y):
        y_xgb = self._encode_survival_labels(y)
        self.model = xgb.XGBRegressor(
            objective="survival:cox",
            eval_metric="cox-nloglik",
            tree_method="hist",
            verbosity=0,
            **self.params,
        )
        self.model.fit(X, y_xgb)
        return self

    def predict_risk(self, X):
        """Return raw risk scores (higher = higher risk)."""
        return self.model.predict(X)

    def score(self, X, y):
        """Returns the Concordance Index (C-index)."""
        risk_scores = self.predict_risk(X)
        events = y["Status"]
        times = y["Time"]
        ci, _, _, _, _ = concordance_index_censored(events, times, risk_scores)
        return ci
