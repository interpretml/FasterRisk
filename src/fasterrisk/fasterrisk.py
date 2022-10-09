import numpy as np
import sklearn.metrics

from fasterrisk.sparseBeamSearch import sparseLogRegModel
from fasterrisk.sparseDiversePool import sparseDiversePoolLogRegModel
from fasterrisk.rounding import starRaySearchModel

from fasterrisk.utils import compute_logisticLoss_from_X_y_beta0_betas, get_all_product_booleans, get_support_indices, isEqual_upTo_8decimal, isEqual_upTo_16decimal, get_all_product_booleans

class RiskScoreOptimizer:
    def __init__(self, X, y, k, select_top_m=50, lb=-5, ub=5, \
                 gap_tolerance=0.05, parent_size=10, child_size=None, \
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
        parent_size : int, optional
            how many solutions to retain after beam search, by default 10
        child_size : _type_, optional
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
        self.parent_size = parent_size
        self.child_size = self.parent_size
        if child_size is not None:
            self.child_size = child_size

        self.sparseDiverseSet_gap_tolerance = gap_tolerance
        self.sparseDiverseSet_select_top_m = select_top_m
        self.sparseDiverseSet_maxAttempts = maxAttempts

        self.sparseLogRegModel_object = sparseLogRegModel(X, y, intercept=True)
        self.sparseDiversePoolLogRegModel_object = sparseDiversePoolLogRegModel(X, y, intercept=True)
        self.starRaySearchModel_object = starRaySearchModel(X = X, y = y, num_ray_search=num_ray_search, early_stop_tolerance=lineSearch_early_stop_tolerance)

    def optimize(self):
        """performs sparseBeamSearch, generates integer sparseDiverseSet, and perform star ray search
        """
        self.sparseLogRegModel_object.get_sparse_sol_via_OMP(k=self.k, parent_size=self.parent_size, child_size=self.child_size)
        
        beta0, betas, ExpyXB = self.sparseLogRegModel_object.get_beta0_betas_ExpyXB()
        self.sparseDiversePoolLogRegModel_object.warm_start_from_beta0_betas_ExpyXB(beta0 = beta0, betas = betas, ExpyXB = ExpyXB)
        sparseDiversePool_beta0, sparseDiversePool_betas = self.sparseDiversePoolLogRegModel_object.get_sparseDiversePool(gap_tolerance=self.sparseDiverseSet_gap_tolerance, select_top_m=self.sparseDiverseSet_select_top_m, maxAttempts=self.sparseDiverseSet_maxAttempts)

        self.multipliers, self.sparseDiversePool_beta0_integer, self.sparseDiversePool_betas_integer = self.starRaySearchModel_object.star_ray_search_scale_and_round(sparseDiversePool_beta0, sparseDiversePool_betas)

    def get_models(self, model_index=None):
        """get risk score models

        Parameters
        ----------
        model_index : int, optional
            index of the model in the integer sparseDiverseSet, by default None

        Returns
        -------
        multipliers : float[:] 
            multipliers with each entry as multipliers[i] 
        sparseDiversePool_integer : float[:, :]
            integer coefficients (intercept included) with each row as an integer solution sparseDiversePool_integer[i]
        """
        if model_index is not None:
            return self.multipliers[model_index], self.sparseDiversePool_beta0_integer[model_index], self.sparseDiversePool_betas_integer[model_index]
        return self.multipliers, self.sparseDiversePool_beta0_integer, self.sparseDiversePool_betas_integer



class RiskScoreClassifier:
    def __init__(self, multiplier, intercept, coefficients, featureNames = None):
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

        self.reset_featureNames(featureNames)

    def predict(self, X):
        """Predict labels

        Parameters
        ----------
        X : float[:, :]
            feature matrix with shape (n, p)

        Returns
        -------
        y_pred : float[:]
            predicted labels (+1.0 or -1.0) with shape (n, )
        """
        y_score = (self.intercept + X.dot(self.coefficients)) / self.multiplier # numpy dot.() has some floating point error issues, so we avoid using self.scaled_intercept and self.scaled_coefficients directly
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
        y_pred_prob : float[:]
            probabilities of each sample y_i to be +1 with shape (n, )
        """
        y_score = (self.intercept + X.dot(self.coefficients)) / self.multiplier # numpy dot.() has some floating point error issues, so we avoid using self.scaled_intercept and self.scaled_coefficients directly
        y_pred_prob = 1/(1+np.exp(-y_score))

        return y_pred_prob

    def compute_logisticLoss(self, X, y):
        """Compute the total logistic loss given the feature matrix X and labels y

        Parameters
        ----------
        X : float[:, :]
            feature matrix with shape (n, p)
        y : float[:]
            sample labels (+1 or -1) with shape (n)

        Returns
        -------
        logisticLoss: float
            total logistic loss, loss = $sum_{i=1}^n log(1+exp(-y_i * (beta0 + X[i, :] @ beta) / multiplier))$
        """
        return compute_logisticLoss_from_X_y_beta0_betas(X, y, self.scaled_intercept, self.scaled_coefficients)

    def get_acc_and_auc(self, X, y):
        """Calculate ACC and AUC of a certain dataset with features X and label y

        Parameters
        ----------
        X : float[:, :]
            2D array storing the features
        y : float[:]
            1D array storing the labels (+1/-1) 

        Returns
        -------
        acc: float
            accuracy
        auc: float
            area under the ROC curve
        """
        y_pred = self.predict(X)
        # print(y_pred.shape, y.shape)
        acc = np.sum(y_pred == y) / len(y)
        y_pred_prob = self.predict_prob(X)

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=y, y_score=y_pred_prob, drop_intermediate=False)
        auc = sklearn.metrics.auc(fpr, tpr)
        return acc, auc

    def reset_featureNames(self, featureNames):
        """Reset the feature names in the class in order to print out the model card for the user

        Parameters
        ----------
        featureNames : str[:]
            a list of strings which are the feature names for columns of X
        """
        self.featureNames = featureNames

    def _print_score_calculation_table(self):
        assert self.featureNames is not None, "please pass the featureNames to the model by using the function .reset_featureNames(featureNames)"

        nonzero_indices = get_support_indices(self.coefficients)

        max_feature_length = max([len(featureName) for featureName in self.featureNames])
        row_score_template = '{0}. {1:>%d}     {2:>2} point(s) | + ...' % (max_feature_length)

        print("The Risk Score is:")
        for count, feature_i in enumerate(nonzero_indices):
            row_score_str = row_score_template.format(count+1, self.featureNames[feature_i], int(self.coefficients[feature_i]))
            if count == 0:
                row_score_str = row_score_str.replace("+", " ")
            
            print(row_score_str)

        final_score_str = ' ' * (14+max_feature_length) + 'SCORE | =    '
        print(final_score_str)

    def _print_score_risk_row(self, scores, risks):
        score_row = "SCORE |"
        risk_row = "RISK  |"
        score_entry_template = '  {0:>4}  |'
        risk_entry_template = ' {0:>5}% |'
        for (score, risk) in zip(scores, risks):
            score_row += score_entry_template.format(score)
            risk_row += risk_entry_template.format(round(100*risk, 1))
        print(score_row)
        print(risk_row)

    def _print_score_risk_table(self):
        nonzero_indices = get_support_indices(self.coefficients)
        all_product_booleans = get_all_product_booleans(len(nonzero_indices))
        all_scores = all_product_booleans.dot(self.coefficients[nonzero_indices])
        all_scores = np.unique(all_scores)
        all_scaled_scores = (self.intercept + all_scores) / self.multiplier
        all_risks = 1 / (1 + np.exp(-all_scaled_scores))

        num_scores_div_2 = (len(all_scores) + 1) // 2
        self._print_score_risk_row(all_scores[:num_scores_div_2], all_risks[:num_scores_div_2])
        self._print_score_risk_row(all_scores[num_scores_div_2:], all_risks[num_scores_div_2:])

    def print_model_card(self):
        """Print the score evaluation table and score risk table onto terminal
        """
        self._print_score_calculation_table()
        self._print_score_risk_table()
        

