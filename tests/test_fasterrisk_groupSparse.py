import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split

from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from fasterrisk.binarization_util import convert_continuous_df_to_binary_df

def get_expected_answers():
    pass

def save_to_dict(int_sols_dict, multiplier, int_sol, train_acc, test_acc, train_auc, test_auc, logisticLoss):
    int_sols_dict["multipliers"].append(multiplier)
    int_sols_dict["int_sols"].append(int_sol)
    int_sols_dict["train_accs"].append(train_acc)
    int_sols_dict["test_accs"].append(test_acc)
    int_sols_dict["train_aucs"].append(train_auc)
    int_sols_dict["test_aucs"].append(test_auc)
    int_sols_dict["logisticLosses"].append(logisticLoss)

def test_check_solutions_interface():
    # import data
    pima_original_data_file_path = "tests/pima_original_data.csv"
    pima_original_data_df = pd.read_csv(pima_original_data_file_path)
    y = np.asarray(pima_original_data_df["Outcome"].values)
    X_original_df = pima_original_data_df.drop(columns="Outcome") # drop the Outcome column, which stores the y label for this binary classification problem

    X_binarized_df, featureIndex_to_groupIndex = convert_continuous_df_to_binary_df(X_original_df, get_featureIndex_to_groupIndex=True)
    X = np.asarray(X_binarized_df)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    lambda2 = 1e-8
    sparsity = 5
    sparseDiversePool_gap_tolerance = 0.05
    sparseDiversePool_select_top_m = 50
    parent_size = 10
    child_size = 10
    maxAttempts = 50
    num_ray_search = 20
    lineSearch_early_stop_tolerance = 0.001 
    group_sparsity = 3

    
    # obtain sparse scoring systems
    int_sols_dict = {"int_sols": [], "train_accs": [], "test_accs": [], "train_aucs": [], "test_aucs": [], "logisticLosses": [], "multipliers": []}
    
    RiskScoreOptimizer_m = RiskScoreOptimizer(X = X_train, y = y_train, k = sparsity, select_top_m = sparseDiversePool_select_top_m, gap_tolerance = sparseDiversePool_gap_tolerance, parent_size = parent_size, maxAttempts = maxAttempts, num_ray_search = num_ray_search, lineSearch_early_stop_tolerance = lineSearch_early_stop_tolerance, group_sparsity = group_sparsity, featureIndex_to_groupIndex = featureIndex_to_groupIndex)

    start_time = time.time()
    
    RiskScoreOptimizer_m.optimize()
    
    int_sols_dict['run_time'] = time.time() - start_time

    multipliers, sparseDiversePool_beta0_integer, sparseDiversePool_betas_integer = RiskScoreOptimizer_m.get_models()

    for i in range(len(multipliers)):
        multiplier = multipliers[i]
        beta0_integer = sparseDiversePool_beta0_integer[i]
        betas_integer = sparseDiversePool_betas_integer[i]

        RiskScoreClassifier_m = RiskScoreClassifier(multiplier, beta0_integer, betas_integer)
        logisticLoss = RiskScoreClassifier_m.compute_logisticLoss(X_train, y_train)
        
        train_acc, train_auc = RiskScoreClassifier_m.get_acc_and_auc(X_train, y_train)
        test_acc, test_auc = RiskScoreClassifier_m.get_acc_and_auc(X_test, y_test)

        integer_sol = np.insert(betas_integer, 0, beta0_integer)
        save_to_dict(int_sols_dict, multiplier, integer_sol, train_acc, test_acc, train_auc, test_auc, logisticLoss)

    # check whether each solution satisfies the group sparsity constraint
    for sol in int_sols_dict['int_sols']:
        sol = sol[1:]
        support = sol.nonzero()[0]
        groupIndices = featureIndex_to_groupIndex[support]
        num_unique_groupIndices = len(np.unique(groupIndices))
        assert num_unique_groupIndices <= group_sparsity, "group sparsity constraint is not satisfied!"
        assert len(support) <= sparsity, "sparsity constraint is not satisfied!"

if __name__ == "__main__":
    test_check_solutions_interface()