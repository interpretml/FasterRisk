import io
from contextlib import redirect_stdout
from typing import *
import warnings

import numpy as np
import pandas as pd
from numpy.typing import *
from PIL import Image
from sklearn.base import BaseEstimator

from fasterrisk.fasterrisk import (RiskScoreClassifier, RiskScoreOptimizer,
                                   get_support_indices)
from fasterrisk.score_visual import (ScoreCardVisualizer, TableVisualizer,
                                     combine_images, output_to_score_intervals,
                                     output_to_score_risk_df)


def shift_coefficients(feature_offsets: pd.DataFrame, features_and_betas: List) -> np.ndarray:
    """
    Shift coefficients to [0, inf) for better visualization, note that predicted risk remain unchanged so doesn't affect model performance.

    Parameters
    ----------
    feature_offsets : pd.DataFrame
        dataframe containing feature with their corresponding offsets
    features_and_betas : List
        list containing features and their corresponding coefficients

    Returns
    -------
    np.ndarray
        shifted coefficients
    """
    features = features_and_betas[0]
    feature_offsets.set_index('feature', inplace = True)
    selected_features = np.asarray(features_and_betas[0])[get_support_indices(features_and_betas[1])]
    selected_features = np.unique([name.split('<=')[0] if 'isNaN' not in name else '' for name in selected_features])
    
    for i in range(len(features)):
        feature = features[i]
        try:
            next_feature = features[i+1]
        except IndexError:
            if feature.split('<=')[0] in selected_features:         # lastly if the last feature is also among selected features, shift it as well
                features_and_betas[1][i] -= feature_offsets.loc[feature.split('<=')[0], 'interval_points']
            break
        
        if 'isNaN' in feature:          # every NaN needs to be 0
            features_and_betas[1][i] = 0
            continue

        # every last feature of selected ones needs to be shifted
        if ('isNaN' in next_feature or feature.split('<=')[0] != next_feature.split('<=')[0]) and feature.split('<=')[0] in selected_features:     # NOTE: changed to include the edge case where isNaN is not available (no missing values in dataset)
            features_and_betas[1][i] -= feature_offsets.loc[feature.split('<=')[0], 'interval_points']
    
    return features_and_betas[1]
        
def get_max_features(X_train: pd.DataFrame) -> Dict[str, float]:
    """
    Get maximum features, helper for checking edge cases in risk score card

    Returns
    -------
    Dict[str, float]
        dictionary keyed by feature name and valued by the maximum value of the feature
    """
    feature_max_dict = {}
    columns = list(X_train.columns)
    for i in range(len(columns)):
        binarized_feature = columns[i]
        try:
            next_binarized_feature = columns[i+1]
        except IndexError:
            split = binarized_feature.split('<=')    
            feature_max_dict[split[0]] = float(split[1])            # if this is the last binarized feature, add
            continue
        if 'isNaN' in binarized_feature:
            continue
        split = binarized_feature.split('<=')

        # if encountered a new feature, this assumes that bins for one feature is altogether
        if 'isNaN' in next_binarized_feature or split[0] != next_binarized_feature.split('<=')[0]:      # NOTE: changed to include the edge case where isNaN is not available (no missing values in dataset)
            feature_max_dict[split[0]] = float(split[1])
    
    return feature_max_dict


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

        assert hasattr(self, 'is_fitted_'), "Please fit the model first"
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

        assert hasattr(self, 'is_fitted_'), "Please fit the model first"
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
        assert hasattr(self, 'is_fitted_'), "Please fit the model first"
        return self.multipliers_, self.beta0_, self.betas_

    def print_risk_card(self, feature_names: List[str], X_train: np.ndarray, y_train: np.ndarray=None, model_idx: int = 0, quantile_len: int=30) -> None:
        """
        print risk score card

        Parameters
        ----------
        feature_names : list
            feature names for the features
        X_train : np.ndarray
        y_train : np.ndarray, optional
        if provided, prints logistic loss on training set
        model_idx : int, optional
            index for the classifier in the pool of solutions, by default 0, which is the classifier with minimum logistic loss
        """
        assert hasattr(self, 'is_fitted_'), "Please fit the model first"
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.to_numpy()
            
        multiplier = self.multipliers_[model_idx]
        beta0 = self.beta0_[model_idx]
        betas = self.betas_[model_idx]

        clf = RiskScoreClassifier(multiplier=multiplier, intercept=beta0, coefficients=betas, X_train=X_train)
        clf.reset_featureNames(feature_names)
        clf.print_model_card(quantile_len=quantile_len)
        if y_train is not None:
            print(f"Logistic loss on training set is {clf.compute_logisticLoss(X_train, y_train)}")

    def visualize_risk_card(
        self, 
        names: List[str],
        X_train: Union[np.ndarray, pd.DataFrame], 
        title: str = 'RISK SCORE CARD', 
        model_idx: Optional[int] = 0, 
        save_path: Optional[str] = None, 
        quantile_len: Optional[int] = 30,
        custom_row_order: Optional[List[str]] = None,
        center_box_width: Optional[int] = 600,
        border_width: Optional[int] = 1,
        ) -> Optional[Image.Image]:
        """
        visualize score card as an image

        Parameters
        ----------
        names : List[str]
            feature names
        X_train : np.ndarray | pd.DataFrame
            training data
        title : str
            title of the entire score card
        save_path : str, optional
            directory to save score card to, by default None
        quantile_len : int, optional
            number of quantiles to use for score to risk conversion table, equivalent to number of cells in risk table, by default 30
        custom_row_order : List[str], optional
            custom order for features in risk score card, by default None
        center_box_width: int, optional
            width of the center box, by default 600
        border_width: int, optional
            width of the border, by default 2

        Returns
        -------
        Optional[Image.Image]
            if save_path is None, return the score card as an image
        """
        assert hasattr(self, 'is_fitted_'), "Please fit the model first"
        
        feature_max_dict = get_max_features(X_train)
        feature_names = list(X_train.columns)

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        
        output = io.StringIO()
        with redirect_stdout(output):
            self.print_risk_card(feature_names = names, X_train = X_train, model_idx = model_idx, quantile_len = quantile_len)
        card = output.getvalue()

        risk_table = output_to_score_risk_df(card)
        score_intervals,  feature_offsets = output_to_score_intervals(card, feature_max_dict = feature_max_dict)
        
        shifted_betas = shift_coefficients(feature_offsets, [feature_names, self.betas_[model_idx].copy()])        # perform shift to scores for better visualization
        shifted_scores = X_train.dot(shifted_betas)
        
        shifted_scores = np.unique(shifted_scores)
        
        quantile_len = min(quantile_len, len(shifted_scores))
        quantile_points = np.asarray(range(1, 1+quantile_len)) / quantile_len
        shifted_scores = np.quantile(shifted_scores, quantile_points, method = 'closest_observation')
        
        try:
            risk_table['Score'] = shifted_scores                        # update scores in risk table
        except ValueError:
            warnings.warn("Unable to update scores in risk table using shifts, please check if quantile_len is too large. Larger training set can generally support larger quantile_len.")

        scv = ScoreCardVisualizer(score_intervals)
        tbv = TableVisualizer(risk_table)

        score_card_img = scv.generate_visual_card(custom_row_order = custom_row_order, center_box_width = center_box_width, border_width = border_width)
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