import numpy as np
import pandas as pd
import time

from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier

def get_expected_answers():
    expected_logisticLosses = np.asarray([9798.65234652, 9859.61575793, 9883.32446183, 9895.72806775, 9914.75974232, 9923.88169028, 9980.63948359, 9988.04100159, 10000.8031389, 10000.83840664, 10004.8263481, 10023.14478069, 10023.50765359, 10024.81637857, 10027.26240921, 10027.26240921, 10028.55771426, 10030.43908377, 10035.51037261, 10045.51694957, 10054.17913732, 10054.90342954, 10057.369164, 10058.18369746, 10064.52587881, 10064.9127764, 10073.9211114, 10136.94958768, 10156.0060037, 10159.61102116, 10184.96322485, 10196.13880181, 10206.19527768, 10226.01355826, 10229.435057, 10232.0743905, 10232.08711344, 10232.08711344, 10232.62547509, 10252.62215612, 10258.24932786, 10261.12578992]) # sorted
    expected_test_accs = np.asarray([0.81787469, 0.81848894, 0.81511057, 0.81342138, 0.80052211, 0.81342138, 0.81234644, 0.81342138, 0.81342138, 0.81464988, 0.81357494, 0.81418919, 0.81572482, 0.81357494, 0.81265356, 0.81265356, 0.81480344, 0.81342138, 0.81296069, 0.81311425, 0.81280713, 0.81342138, 0.81342138, 0.81449631, 0.81342138, 0.81403563, 0.81449631, 0.81449631, 0.81449631, 0.817414, 0.81449631, 0.81449631, 0.81449631, 0.81449631, 0.81449631, 0.81449631, 0.81449631, 0.81449631, 0.81664619, 0.81449631, 0.81418919, 0.81449631]) # sorted
    expected_test_aucs = np.asarray([0.85636652, 0.85375202, 0.85423695, 0.85606471, 0.85639657, 0.85201393, 0.84940344, 0.84900004, 0.848115, 0.85166874, 0.84651199, 0.85057216, 0.84776618, 0.84619568, 0.85057035, 0.85057035, 0.8463334, 0.84996381, 0.84969807, 0.84548625, 0.84652209, 0.84869576, 0.84579777, 0.84886179, 0.84491274, 0.84870619, 0.84846415, 0.84784233, 0.84384685, 0.843995, 0.84378728, 0.84443765, 0.84306587, 0.84235962, 0.84185992, 0.84184879, 0.84461416, 0.84461416, 0.84387152, 0.84289305, 0.84161199, 0.841296]) # sorted
    expected_multipliers = np.asarray([1.6009213, 1.63023211, 1.9506044, 1.90704887, 1.80838131, 1.8649294, 1.22922891, 1.35263814, 1.37290238, 1.61114757, 1.4386743, 1.85528532, 1.54384815, 1.46150212, 1.69941044, 1.69941044, 1.46236241, 1.91678996, 1.53613056, 1.44120224, 1.56952033, 1.54257895, 1.4856219, 1.64627032, 1.44689776, 1.82633662, 1.38555742, 1.66478981, 1.48093683, 1.74049054, 1.69736044, 1.43394088, 1.6909653, 1.59960591, 1.8414886, 1.60098164, 1.79963122, 1.79963122, 1.43593351, 1.77281803, 1.48820364, 1.40751051]) # sorted

    return expected_logisticLosses, expected_test_accs, expected_test_aucs, expected_multipliers

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
    train_data = np.asarray(pd.read_csv("tests/adult_train_data.csv"))
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    test_data = np.asarray(pd.read_csv("tests/adult_test_data.csv"))
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
    
    
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

    int_sols_dict["logisticLosses"] = np.asarray(int_sols_dict["logisticLosses"])
    int_sols_dict["test_accs"] = np.asarray(int_sols_dict["test_accs"])
    int_sols_dict["test_aucs"] = np.asarray(int_sols_dict["test_aucs"])
    int_sols_dict["multipliers"] = np.asarray(int_sols_dict["multipliers"])

    # check the answers
    expected_logisticLosses, expected_test_accs, expected_test_aucs, expected_multipliers = get_expected_answers()
  
    assert len(int_sols_dict["logisticLosses"]) == len(expected_logisticLosses), "logistcLosses do not have the expected length"
    assert np.max(np.abs(int_sols_dict["logisticLosses"] - expected_logisticLosses)) < 1e-8, "logisticLosses values are not correct"
    assert np.max(np.abs(int_sols_dict["multipliers"] - expected_multipliers)) < 1e-8, "multipliers values are not correct"
    assert np.max(np.abs(int_sols_dict["test_accs"] - expected_test_accs)) < 1e-8, "test_accs values are not correct"
    assert np.max(np.abs(int_sols_dict["test_aucs"] - expected_test_aucs)) < 1e-8, "test_aucs values are not correct"

if __name__ == "__main__":
    test_check_solutions_interface()