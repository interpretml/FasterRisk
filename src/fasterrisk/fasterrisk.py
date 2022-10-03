import numpy as np

from fasterrisk.bounded_sparse_level_set import sparseLogRegModel
from fasterrisk.heuristic_rounding import starRaySearchModel
from fasterrisk.utils import compute_logisticLoss_from_X_y_beta0_betas

class RiskScoreOptimizer:
    def __init__(self, X, y, k, select_top_m=50, lb=-5, ub=5, \
                 gap_tolerance=0.05, beam_size=10, sub_beam_size=None, \
                 maxAttempts=50, num_ray_search=20, \
                 lineSearch_early_stop_tolerance=0.001):

        # check the formats of inputs X and y
        y_shape = y.shape
        y_unique = np.unique(y)
        y_unique_expected = np.asarray([-1, 1])
        X_shape = X.shape
        print("y_shape is ", y_shape)
        assert len(y_shape) == 1, "input y must have 1-D shape!"
        assert len(y_unique) == 2, "input y must have only 2 labels"
        assert max(np.abs(y_unique - y_unique_expected)) < 1e-8, "input y must be equal to only +1 or -1"
        assert len(X_shape) == 2, "input X must have 2-D shape!"
        assert X_shape[0] == y_shape[0], "number of rows from input X must be equal to the number of elements from input y!"

        self.k = k
        self.beam_size = beam_size
        self.sub_beam_size = self.beam_size
        if sub_beam_size is not None:
            self.sub_beam_size = sub_beam_size

        self.sparseDiverseSet_gap_tolerance = gap_tolerance
        self.sparseDiverseSet_select_top_m = select_top_m
        self.sparseDiverseSet_maxAttempts = maxAttempts

        self.sparseLogRegModel_object = sparseLogRegModel(X, y, intercept=True)

        self.starRaySearchModel_object = starRaySearchModel(X = X, y = y, num_ray_search=num_ray_search, early_stop_tolerance=lineSearch_early_stop_tolerance)

    def optimize(self):
        self.sparseLogRegModel_object.get_sparse_sol_via_OMP(k=self.k, beam_size=self.beam_size, sub_beam_size=self.sub_beam_size)
        
        sparse_diverse_set_continuous = self.sparseLogRegModel_object.get_sparse_diverse_set(gap_tolerance=self.sparseDiverseSet_gap_tolerance, select_top_m=self.sparseDiverseSet_select_top_m, maxAttempts=self.sparseDiverseSet_maxAttempts)

        self.multipliers, self.sparse_diverse_set_integer = self.starRaySearchModel_object.star_ray_search_scale_and_round(sparse_diverse_set_continuous)

    def get_models(self, model_index=None):
        if model_index is not None:
            return self.multipliers[model_index], self.sparse_diverse_set_integer[:, model_index]
        return self.multipliers, self.sparse_diverse_set_integer

    def print_model_card(self, model_index=None):
        pass


class RiskScoreClassifier:
    def __init__(self, multiplier, intercept, coefficients):
        self.multiplier = multiplier
        self.intercept = intercept
        self.coefficients = coefficients

        self.scaled_intercept = self.intercept / self.multiplier
        self.scaled_coefficients = self.coefficients / self.multiplier

    def predict(self, X):
        y_score = self.scaled_intercept + X.dot(self.scaled_coefficients)
        y_pred = 2 * y_score - 1
        return y_pred

    def predict_prob(self, X):
        y_score = self.scaled_intercept + X.dot(self.scaled_coefficients)
        y_pred_prob = 1/(1+np.exp(-y_score))
        return y_pred_prob

    def compute_logisticLoss(self, X, y):
        return compute_logisticLoss_from_X_y_beta0_betas(X, y, self.scaled_intercept, self.scaled_coefficients)