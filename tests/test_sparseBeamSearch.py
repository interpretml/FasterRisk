import numpy as np
import pandas as pd
from fasterrisk.sparseBeamSearch import sparseLogRegModel
from fasterrisk.utils import isEqual_upTo_8decimal

def get_expected_beta0_betas():
    expected_beta0 = -1.8121419975067676
    expected_betas = np.zeros(36)
    expected_betas[1] = -79.10534673
    expected_betas[9] = -109.46042181
    expected_betas[13] = -149.58867835
    expected_betas[20] = 190.72096681
    expected_betas[34] = 74.72272035
    return expected_beta0, expected_betas

def test_sparseBeamSearch():
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
    
    
    sparseLogRegModel_object = sparseLogRegModel(X_train, y_train, intercept=True)

    sparseLogRegModel_object.get_sparse_sol_via_OMP(k=sparsity, parent_size=parent_size, child_size=parent_size)

    beta0, betas, ExpyXB = sparseLogRegModel_object.get_beta0_betas_ExpyXB()

    expected_beta0, expected_betas = get_expected_beta0_betas()
    
    assert isEqual_upTo_8decimal(expected_beta0, beta0), "beta0 produced by sparseBeamSearch is not correct"
    assert isEqual_upTo_8decimal(expected_betas, betas), "betas produced by sparseBeamSearch is not correct"
     
    # print(beta0)
    # nonzero_indices = np.where(np.abs(betas) > 1e-8)[0]
    # print(nonzero_indices)
    # print(betas[nonzero_indices])
    # print(len(betas))

if __name__ == '__main__':
    test_sparseBeamSearch()