import econml.grf
from sklearn.linear_model import LinearRegression, LogisticRegression
from econml.ortho_forest import DMLOrthoForest, DROrthoForest
# from econml.utilities import _RegressionWrapper, WeightedModelWrapper

from causal_estimators.base import BaseEconMLEstimator



# class RegWrapper:
#     def __init__(self, classifier):
#         self.classifier = classifier
#     def fit(self, X, y, **kwargs):
#         return self.classifier.fit(X, y, **kwargs)
#     def predict(self, X):
#         return self.classifier.predict_proba(X)[:, 1]


class ORthoforestDML(BaseEconMLEstimator):

    def __init__(self, outcome_model,
                 prop_score_model=LogisticRegression(),
                 n_trees = 100, max_depth = 10, subsample_ratio = 0.7, lambda_reg = 1,
                 discrete_treatment=True, min_leaf_size=200):
        # TODO: add other options that NonParamDMLCateEstimator allows?
        super().__init__(econml_estimator=DMLOrthoForest(min_leaf_size=min_leaf_size,
            model_Y=outcome_model, model_T=prop_score_model, model_Y_final=outcome_model,
            discrete_treatment=discrete_treatment,n_trees=n_trees, max_depth=max_depth,
                                                         subsample_ratio=subsample_ratio,
                                                         lambda_reg=lambda_reg))


class CausalForest(BaseEconMLEstimator):

    def __init__(self, criterion='mse', n_estimators=100, min_samples_leaf=5, max_depth=None,
                       min_var_fraction_leaf=None, min_var_leaf_on_val=False,
                       min_impurity_decrease = 0.0, max_samples=0.45, min_balancedness_tol=0.45,
                       warm_start=False, inference=True, fit_intercept=True, subforest_size=4,
                       honest=True, verbose=0, n_jobs=-1, random_state=1235):
        # TODO: add other options that NonParamDMLCateEstimator allows?
        super().__init__(econml_estimator=econml.grf.CausalForest(criterion=criterion, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                       min_var_fraction_leaf=None, min_var_leaf_on_val=True,
                       min_impurity_decrease = min_impurity_decrease, max_samples=max_samples, min_balancedness_tol=min_balancedness_tol,
                       warm_start=warm_start, inference=inference, fit_intercept=fit_intercept, subforest_size=subforest_size,
                       honest=honest, verbose=verbose, n_jobs=n_jobs, random_state=random_state))

    def fit(self, w, t, y):
        self.forest_fit(w, t, y)

    def estimate_ate(self, w):
        return self.estimate_ate_forest(w)

    def estimate_ite(self, w):
        return self.estimate_ite_forest(w)


class CausalTree(BaseEconMLEstimator):

    def __init__(self, criterion='het', n_estimators=1, min_samples_leaf=10, max_depth=5,
                       min_var_fraction_leaf=None, min_var_leaf_on_val=True,
                       min_impurity_decrease = 0.0, max_samples=0.45, min_balancedness_tol=0.45,
                       warm_start=False, inference=False, fit_intercept=True, subforest_size=1,
                       honest=True, verbose=0, n_jobs=-1, random_state=1235):
        # TODO: add other options that NonParamDMLCateEstimator allows?
        super().__init__(econml_estimator=econml.grf.CausalForest(criterion=criterion, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                       min_var_fraction_leaf=None, min_var_leaf_on_val=True,
                       min_impurity_decrease = min_impurity_decrease, max_samples=max_samples, min_balancedness_tol=min_balancedness_tol,
                       warm_start=warm_start, inference=inference, fit_intercept=fit_intercept, subforest_size=subforest_size,
                       honest=honest, verbose=verbose, n_jobs=n_jobs, random_state=random_state))