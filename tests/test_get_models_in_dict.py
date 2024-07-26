import numpy as np
import pandas as pd
import time

from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from fasterrisk.utils import get_groupIndex_from_featureNames

import sys


def get_models_in_dict(data_path, y_label_name):
    df = pd.read_csv(data_path)
    X_df = df.drop(columns = [y_label_name])
    # print(df.head())
    # print(X_df.head())

    X_featureNames = list(X_df.columns)

    train_data = np.asarray(df)
    X_train, y_train = train_data[:, 1:], train_data[:, 0]

    # convert y to -1, 1
    y_unique = np.sort(np.unique(y_train))
    if np.any(y_unique != np.array([-1, 1])):
        y_train = 2 * (y_train - y_unique[0]) / (y_unique[1] - y_unique[0]) - 1


    lambda2 = 1e-8
    sparsity = 5
    sparseDiversePool_gap_tolerance = 0.05
    sparseDiversePool_select_top_m = 50
    parent_size = 10
    child_size = 10
    maxAttempts = 50
    num_ray_search = 20
    lineSearch_early_stop_tolerance = 0.001 

    # obtain sparse scoring systems
    int_sols_dict = {"int_sols": [], "train_accs": [], "test_accs": [], "train_aucs": [], "test_aucs": [], "logisticLosses": [], "multipliers": []}

    RiskScoreOptimizer_m = RiskScoreOptimizer(X = X_train, y = y_train, k = sparsity, select_top_m = sparseDiversePool_select_top_m, gap_tolerance = sparseDiversePool_gap_tolerance, parent_size = parent_size, maxAttempts = maxAttempts, num_ray_search = num_ray_search, lineSearch_early_stop_tolerance = lineSearch_early_stop_tolerance)

    start_time = time.time()

    RiskScoreOptimizer_m.optimize()

    int_sols_dict['run_time'] = time.time() - start_time

    featureIndex_to_groupIndex = get_groupIndex_from_featureNames(X_featureNames)
    model_dict_list = RiskScoreOptimizer_m.get_models_in_dict(featureNames=X_featureNames, featureIndex_to_groupIndex=featureIndex_to_groupIndex)

    # print(model_dict_list)
    return model_dict_list

def test_get_models_in_dict_adult():
    data_path = "tests/adult_train_data.csv"
    y_label_name = "Over50K"
    models_in_dict = get_models_in_dict(data_path, y_label_name)

    rng = np.random.default_rng(0)
    random_model = rng.choice(models_in_dict, 1)[0]

    # print(random_model)
    first_feature = random_model['feature_data'][0]
    training_logistic_loss = 10232.07439049685
    training_accuracy = 0.818112019655265
    training_AUC = 0.8481360500257804
    card_label = 35

    assert first_feature[0] == -2.0, "First feature coefficient is not -2.0"
    assert first_feature[1] == "Age_22_to_29", "First feature name is not Age_22_to_29"
    assert random_model['training_logistic_loss'] == training_logistic_loss, "Training logistic loss is not correct"
    assert random_model['training_accuracy'] == training_accuracy, "Training accuracy is not correct"
    assert random_model['training_AUC'] == training_AUC, "Training AUC is not correct"
    assert random_model['card_label'] == card_label, "Card label is not correct"

def test_get_models_in_dict_fico():
    data_path = "tests/fico_data.csv"
    y_label_name = "RiskPerformance"
    models_in_dict = get_models_in_dict(data_path, y_label_name)

    rng = np.random.default_rng(0)
    random_model = rng.choice(models_in_dict, 1)[0]

    # print(random_model)
    first_feature = random_model['feature_data'][0]
    training_logistic_loss = 5958.476051962211
    training_accuracy = 0.7117315230901616
    training_AUC = 0.7726817732185381
    card_label = 42

    assert first_feature[0] == 4.0, "First feature coefficient is not 4.0"
    assert first_feature[1] == "ExternalRiskEstimate<=70", "First feature name is not ExternalRiskEstimate<=70"
    assert random_model['training_logistic_loss'] == training_logistic_loss, "Training logistic loss is not correct"
    assert random_model['training_accuracy'] == training_accuracy, "Training accuracy is not correct"
    assert random_model['training_AUC'] == training_AUC, "Training AUC is not correct"
    assert random_model['card_label'] == card_label, "Card label is not correct"

if __name__ == "__main__":
    test_get_models_in_dict_adult()
    test_get_models_in_dict_fico()