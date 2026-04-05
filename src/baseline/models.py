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
    pass


class XGBoostSurvivalModel:
    pass
