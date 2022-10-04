import numpy as np

from fasterrisk.bounded_sparse_level_set import sparseLogRegModel
from fasterrisk.sparseDiversePoolModel import sparseDiversePoolLogRegModel
from fasterrisk.rounding import starRaySearchModel
from fasterrisk.utils import compute_logisticLoss_from_X_y_beta0_betas

class RiskScoreOptimizer:
    def __init__(self, X, y, k, select_top_m=50, lb=-5, ub=5, \
                 gap_tolerance=0.05, beam_size=10, sub_beam_size=None, \
                 maxAttempts=50, num_ray_search=20, \
                 lineSearch_early_stop_tolerance=0.001):
        """Initialize the RiskScoreOptimizer class, which performs sparseBeamSearch and generates integer sparseDiverseSet

        Parameters
        ----------
        X : float[:, :]
            feature matrix, each row[i, :] corresponds to the features of sample i
        y : float[:]
            labels (+1 or -1) of each sample
        k : int
            number of selected features in the final sparse model
        select_top_m : int, optional
            _description_, by default 50
        lb : float, optional
            lower bound of the coefficients, by default -5
        ub : float, optional
            upper bound of the coefficients, by default 5
        beam_size : int, optional
            how many solutions to retain after beam search, by default 10
        sub_beam_size : _type_, optional
            how many new solutions to expand for each existing solution, by default None
        maxAttempts : int, optional
            how many alternative features to try in order to replace the old feature during the diverse set pool generation, by default None
        num_ray_search : int, optional
            how many multipliers to try for each continuous sparse solution, by default 20
        lineSearch_early_stop_tolerance : float, optional
            tolerance level to stop linesearch early (error_of_loss_difference/loss_of_continuous_solution), by default 0.001
        """

        # check the formats of inputs X and y
        y_shape = y.shape
        y_unique = np.unique(y)
        y_unique_expected = np.asarray([-1, 1])
        X_shape = X.shape
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
        self.sparseDiversePoolLogRegModel_object = sparseDiversePoolLogRegModel(X, y, intercept=True)
        self.starRaySearchModel_object = starRaySearchModel(X = X, y = y, num_ray_search=num_ray_search, early_stop_tolerance=lineSearch_early_stop_tolerance)

    def optimize(self):
        """performs sparseBeamSearch and generates integer sparseDiverseSet
        """
        self.sparseLogRegModel_object.get_sparse_sol_via_OMP(k=self.k, beam_size=self.beam_size, sub_beam_size=self.sub_beam_size)
        
        beta0, betas = self.sparseLogRegModel_object.get_beta0_and_betas()
        self.sparseDiversePoolLogRegModel_object.warm_start_from_beta0_betas(beta0 = beta0, betas = betas)
        sparse_diverse_set_continuous = self.sparseDiversePoolLogRegModel_object.get_sparse_diverse_set(gap_tolerance=self.sparseDiverseSet_gap_tolerance, select_top_m=self.sparseDiverseSet_select_top_m, maxAttempts=self.sparseDiverseSet_maxAttempts)

        self.multipliers, self.sparse_diverse_set_integer = self.starRaySearchModel_object.star_ray_search_scale_and_round(sparse_diverse_set_continuous)

    def get_models(self, model_index=None):
        """get risk score models

        Parameters
        ----------
        model_index : int, optional
            index of the model in the integer sparseDiverseSet, by default None

        Returns
        -------
        multipliers, sparse_diverse_set_integer
            For risk score model i, we can access its multiplier with multipliers[i] and its integer coefficients (intercept included) as sparse_diverse_set_integer[i]
        """
        if model_index is not None:
            return self.multipliers[model_index], self.sparse_diverse_set_integer[:, model_index]
        return self.multipliers, self.sparse_diverse_set_integer

    def print_model_card(self, model_index=None):
        pass


class RiskScoreClassifier:
    def __init__(self, multiplier, intercept, coefficients):
        """Initialize a risk score classifier. Then we can use this classifier to predict labels, predict probabilites, and calculate total logistic loss

        Parameters
        ----------
        multiplier : float
            multiplier of the risk score model
        intercept : float
            intercept of the risk score model
        coefficients : float[:]
            coefficients of the risk score model
        """
        self.multiplier = multiplier
        self.intercept = intercept
        self.coefficients = coefficients

        self.scaled_intercept = self.intercept / self.multiplier
        self.scaled_coefficients = self.coefficients / self.multiplier

    def predict(self, X):
        """Predict labels

        Parameters
        ----------
        X : float[:, :]
            feature matrix with shape (n, p)

        Returns
        -------
        float[:]
            predicted labels (+1.0 or -1.0) with shape (n, )
        """
        y_score = self.scaled_intercept + X.dot(self.scaled_coefficients)
        y_pred = 2 * (y_score > 0) - 1
        return y_pred

    def predict_prob(self, X):
        """Calculate the risk probabilities of predicting each sample y_i with label +1

        Parameters
        ----------
        X : float[:, :]
            feature matrix with shape (n, p)

        Returns
        -------
        float[:]
            probabilities of each sample y_i to be +1 with shape (n, )
        """
        y_score = self.scaled_intercept + X.dot(self.scaled_coefficients)
        y_pred_prob = 1/(1+np.exp(-y_score))
        return y_pred_prob

    def compute_logisticLoss(self, X, y):
        """Compute the total logistic loss given feature matrix and labels

        Parameters
        ----------
        X : float[:, :]
            feature matrix with shape (n, p)
        y : float[:]
            sample labels (+1 or -1) with shape (n)

        Returns
        -------
        float
            total logistic loss, loss = $\sum_{i=1}^n log(1+exp(-y_i * (beta0 + X[i, :] @ beta) / multiplier))$
        """
        return compute_logisticLoss_from_X_y_beta0_betas(X, y, self.scaled_intercept, self.scaled_coefficients)