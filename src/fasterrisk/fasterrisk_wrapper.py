import io
import sys
from typing import *

from fasterrisk.score_visual import output_to_score_intervals, output_to_score_risk_df, TableVisualizer, ScoreCardVisualizer, combine_images
import numpy as np
import pandas as pd
from fasterrisk.fasterrisk import RiskScoreClassifier, RiskScoreOptimizer
from numpy.typing import *
from PIL import Image
from sklearn.base import BaseEstimator
from contextlib import redirect_stdout


class FasterRisk(BaseEstimator):
    """
    Wrapper for FasterRisk algorithm

    Parameters
    ----------
    k : int
        sparsity constraint, equivalent to number of selected (binarized) features in the final sparse model(s)
    select_top_m : int, optional
        number of top solutions to keep among the pool of diverse sparse solutions, by default 50
    lb : float, optional
        lower bound of the coefficients, by default -5
    ub : float, optional
        upper bound of the coefficients, by default 5
    gap_tolerance : float, optional
        tolerance in logistic loss for creating diverse sparse solutions, by default 0.05
    parent_size : int, optional
        how many solutions to retain after beam search, by default 10
    child_size : int, optional
        how many new solutions to expand for each existing solution, by default None
    maxAttempts : int, optional
        how many alternative features to try in order to replace the old feature during the diverse set pool generation, by default None
    num_ray_search : int, optional
        how many multipliers to try for each continuous sparse solution, by default 20
    lineSearch_early_stop_tolerance : float, optional
        tolerance level to stop linesearch early (error_of_loss_difference/loss_of_continuous_solution), by default 0.001
    group_sparsity : int, optional
        number of groups to be selected, by default None
    featureIndex_to_groupIndex : ndarray, optional
        (1D array with `int` type) featureIndex_to_groupIndex[i] is the group index of feature i, by default None
    
    Attributes
    ----------
    multipliers_ : List
        multipliers used in the final diverse sparse models
    beta0_ : List
        intercepts used in the final diverse sparse models
    betas_ : List
        coefficients used in the final diverse sparse models
    """
    def __init__(self, k: int=10, select_top_m: int=50, lb: float=-5, ub: float=5,
                 gap_tolerance: float=0.05, parent_size: int=10, child_size: int=None,
                 maxAttempts: int=50, num_ray_search: int=20,
                 lineSearch_early_stop_tolerance: float=0.001,
                 group_sparsity: int=None,
                 featureIndex_to_groupIndex: np.ndarray=None) -> None:
        self.k = k
        self.select_top_m = select_top_m
        self.lb = lb
        self.ub = ub
        self.gap_tolerance = gap_tolerance
        self.parent_size = parent_size
        self.child_size = child_size
        self.maxAttempts = maxAttempts
        self.num_ray_search = num_ray_search
        self.lineSearch_early_stop_tolerance = lineSearch_early_stop_tolerance
        self.group_sparsity = group_sparsity
        self.featureIndex_to_groupIndex = featureIndex_to_groupIndex

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train FasterRisk

        Parameters
        ----------
        X : np.ndarray
            training data
        y : np.ndarray
            training data labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()
            
        opt = RiskScoreOptimizer(
            X=X, y=y,
            k=self.k, select_top_m=self.select_top_m,
            lb=self.lb, ub=self.ub,
            gap_tolerance=self.gap_tolerance, parent_size=self.parent_size, child_size=self.child_size,
            maxAttempts=self.maxAttempts, num_ray_search=self.num_ray_search,
            lineSearch_early_stop_tolerance=self.lineSearch_early_stop_tolerance,
            group_sparsity=self.group_sparsity,
            featureIndex_to_groupIndex=self.featureIndex_to_groupIndex,
        )
        opt.optimize()  # train
        self.multipliers_, self.beta0_, self.betas_ = opt.get_models()  # obtain trained coefficients
        self.is_fitted_ = True

        return self

    def predict_proba(self, X: np.ndarray, model_idx: int = 0) -> np.ndarray[float]:
        """
        make probability predictions for binary classification

        Parameters
        ----------
        X : np.ndarray
            input data
        model_idx : int, optional
            used to specify which model to use (ranked by increasing order of logistic loss) among diverse sparse models, by default 0, which is the model with minimum logistic loss

        Returns
        -------
        np.ndarray[float]
            probability predictions for binary classification
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        assert self.is_fitted_, "Please fit the model first"
        multiplier = self.multipliers_[model_idx]
        beta0 = self.beta0_[model_idx]
        betas = self.betas_[model_idx]

        y_prob = RiskScoreClassifier(multiplier=multiplier, intercept=beta0, coefficients=betas).predict_prob(X)

        return y_prob

    def predict(self, X: np.ndarray, model_idx: int = 0) -> np.ndarray[int]:
        """
        make bianry prediction

        Parameters
        ----------
        X : np.ndarray
            input data
        model_idx : int, optional
            used to specify which model to use (ranked by increasing order of logistic loss) among diverse sparse models, by default 0, which is the model with minimum logistic loss

        Returns
        -------
        np.ndarray[float]
            binary predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        assert self.is_fitted_, "Please fit the model first"
        multiplier = self.multipliers_[model_idx]
        beta0 = self.beta0_[model_idx]
        betas = self.betas_[model_idx]

        y_pred = RiskScoreClassifier(multiplier=multiplier, intercept=beta0, coefficients=betas).predict(X)

        return y_pred

    def get_model_params(self) -> Tuple[List[float], List[float], List[float]]:
        """
        Get model parameters for FasterRisk

        Returns
        -------
        Three lists of multipliers, beta0s (intercepts), and betas (coefficients)
        """
        return self.multipliers_, self.beta0_, self.betas_

    def print_risk_card(self, feature_names: List[str], X_train: np.ndarray, model_idx: int = 0) -> None:
        """
        print risk score card

        Parameters
        ----------
        feature_names : list
            feature names for the features
        X_train : pd.DataFrame
            training data
        model_idx : int, optional
            index for the classifier in the pool of solutions, by default 0, which is the classifier with minimum logistic loss
        """
        assert self.is_fitted_, "Please fit the model first"
        multiplier = self.multipliers_[model_idx]
        beta0 = self.beta0_[model_idx]
        betas = self.betas_[model_idx]

        clf = RiskScoreClassifier(multiplier=multiplier, intercept=beta0, coefficients=betas, X_train=X_train)
        clf.reset_featureNames(feature_names)
        clf.print_model_card()

    def print_topK_risk_cards(self, feature_names: List[str], X_train: np.ndarray, y_train: np.ndarray, K: int = 10) -> None:
        """
        print top K risk score cards ranked by logistic loss

        Parameters
        ----------
        feature_names : list
            feature names for features
        X_train : np.ndarray
        y_train : np.ndarray
        K : int, optional
            number of cards to print for, by default 10
        """
        assert self.is_fitted_, "Please fit the model first"
        num_models = min(K, len(self.multipliers_))

        for model_index in range(num_models):
            print(f"{'=' * 22} TOP {model_index + 1} SCORE CARD {'=' * 22}")
            multiplier = self.multipliers_[model_index]
            beta0 = self.beta0_[model_index]
            betas = self.betas_[model_index]

            tmp_classifier = RiskScoreClassifier(multiplier, beta0, betas, X_train=X_train)
            tmp_classifier.reset_featureNames(feature_names)
            tmp_classifier.print_model_card()

            train_loss = tmp_classifier.compute_logisticLoss(X_train, y_train)
            print("The logistic loss on the training set is {}".format(train_loss))
    
    def visualize_risk_card(self, names, X_train, score_card_name: str, title: str, model_idx: int=0, save_path: str=None, card: str=None) -> Optional[Image.Image]:
        """
        visualize score card as an image

        Parameters
        ----------
        names : list
            feature names
        X_train : np.ndarray
            training data
        score_card_name : str
            name of score card
        title : str
            title of the entire score card
        save_path : str, optional
            directory to save score card to, by default None
        card : str, optional
            string of printed score card obtained from self.print_risk_card(), by default None

        Returns
        -------
        Optional[Image.Image]
            if save_path is None, return the score card as an image
        """
        if card is None:
            output = io.StringIO()
            with redirect_stdout(output):
                self.print_risk_card(names, X_train, model_idx)
            card = output.getvalue()
        
        risk_table = output_to_score_risk_df(card)
        score_intervals, offset = output_to_score_intervals(card)

        scv = ScoreCardVisualizer(score_intervals, offset)
        tbv = TableVisualizer(risk_table, offset)

        score_card_img = scv.generate_visual_card(score_card_name, custom_row_order=None)
        risk_table_img = tbv.generate_table(title)
        card_img = combine_images(score_card_img, risk_table_img)
        
        if save_path is not None:
            card_img.save(save_path)
        else:
            return card_img
    
    @staticmethod
    def define_bounds(X: pd.DataFrame, feature_bound_pairs: Dict[str, Tuple[Union[float, int], Union[float, int]]], lb_else: Union[float, int], ub_else: Union[float, int]) -> Tuple[List, List]:
        """
        Obtain user defined bounds for each feature in X.

        Parameters
        ----------
        X : pd.DataFrame
            data of interest
        feature_bound_pairs : Dict
            dictionary of feature names and their corresponding bounds, such as {'feature_name': (lb, ub), ...}
        lb_else : Union[float, int]
            lb bounds for all other features not specified by feature_bound_pairs, if all features are included in feature_bound_pairs, then this is lb for the intercept
        ub_else : Union[float, int]
            ub bounds for all other features not specified by feature_bound_pairs, if all features are included in feature_bound_pairs, then this is up for the intercept

        Returns
        -------
        Tuple[List, List]
            lb_bounds, ub_bounds
        """
        assert isinstance(X, pd.DataFrame), "X must be a pandas DataFrame!"
        assert set(list(feature_bound_pairs.keys())).issubset(set(list(X.columns))), "features specified in feature_bound_pairs are not in X!"
        lb_bounds, ub_bounds,  = [], []
        for col in X.columns:
            try:            # if specified in feature_bound_pairs, append the specified bounds
                lb_bounds.append(feature_bound_pairs[col][0])
            except KeyError:         # else, append the default bounds
                lb_bounds.append(lb_else)
            
            try:
                ub_bounds.append(feature_bound_pairs[col][1])
            except KeyError:
                ub_bounds.append(ub_else)
        
        return lb_bounds, ub_bounds