from sklearn.linear_model import LinearRegression, LogisticRegression
from econml.ortho_forest import DMLOrthoForest, DROrthoForest
from econml.utilities import _RegressionWrapper, WeightedModelWrapper

from causal_estimators.base import BaseEconMLEstimator



class RegWrapper:
    def __init__(self, classifier):
        self.classifier = classifier
    def fit(self, X, y, **kwargs):
        return self.classifier.fit(X, y, **kwargs)
    def predict(self, X):
        return self.classifier.predict_proba(X)[:, 1]


class ORthoforestDML(BaseEconMLEstimator):

    def __init__(self, outcome_model,
                 prop_score_model=LogisticRegression(),
                 discrete_treatment=True, min_leaf_size=100):
        # TODO: add other options that NonParamDMLCateEstimator allows?
        super().__init__(econml_estimator=DMLOrthoForest(min_leaf_size=min_leaf_size,
            model_Y=_RegressionWrapper(outcome_model), model_T=prop_score_model, model_Y_final=_RegressionWrapper(outcome_model),
            discrete_treatment=discrete_treatment))