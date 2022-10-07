import numpy as np
from fasterrisk.sparseBeamSearch import sparseLogRegModel
from fasterrisk.sparseDiversePool import sparseDiversePoolLogRegModel
from fasterrisk.utils import isEqual_upTo_8decimal

def get_expected_last_5_solutions():
    expected_last_5_solutions = np.zeros((37, 5))

    expected_last_5_solutions[:, 0][np.asarray([0, 2, 10, 14, 21, 28], dtype=int)] = np.asarray([-1.68640248, -1.24434331, -1.39870223, -2.77834699,  2.22334498, 0.3615457 ])
    expected_last_5_solutions[:, 1][np.asarray([0, 2, 10, 14, 21, 34], dtype=int)] = np.asarray([-1.47609794, -1.25201535, -1.41247325, -2.7125841 ,  2.37800035, -0.51648603])
    expected_last_5_solutions[:, 2][np.asarray([0, 2, 10, 14, 21, 30], dtype=int)] = np.asarray([-1.86056571, -1.25398798, -1.39427183, -2.75367946,  2.34725272,  0.4100869 ])
    expected_last_5_solutions[:, 3][np.asarray([0, 2, 10, 14, 21, 24], dtype=int)] = np.asarray([-1.24597583, -1.18031056, -1.3953493 , -2.75671816,  2.09374404, -0.53623595])
    expected_last_5_solutions[:, 4][np.asarray([0, 2, 10, 14, 21, 22], dtype=int)] = np.asarray([-1.72065552, -1.19777292, -1.39309149, -2.75388333,  2.56853701,  0.48178379])
    return expected_last_5_solutions

def test_sparseDiversePool():
    # import data
    train_test_data = np.load("tests/train_test_data.npz", allow_pickle=True)
    
    X_train, y_train, X_test, y_test = train_test_data["X_train"], train_test_data["y_train"], train_test_data["X_test"], train_test_data["y_test"] 
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    
    lambda2 = 1e-8
    sparsity = 5
    sparseLevelSet_gap_tolerance = 0.05
    sparseLevelSet_select_top_m = 50
    parent_size = 10
    child_size = 10
    maxAttempts = 50
    num_ray_search = 20
    lineSearch_early_stop_tolerance = 0.001 
    
    
    sparseLogRegModel_object = sparseLogRegModel(X_train, y_train, intercept=True)

    sparseLogRegModel_object.get_sparse_sol_via_OMP(k=sparsity, parent_size=parent_size, child_size=parent_size)

    beta0, betas, ExpyXB = sparseLogRegModel_object.get_beta0_betas_ExpyXB()


    sparseDiversePoolLogRegModel_object = sparseDiversePoolLogRegModel(X_train, y_train, intercept=True)


    sparseDiversePoolLogRegModel_object.warm_start_from_beta0_betas_ExpyXB(beta0 = beta0, betas = betas, ExpyXB = ExpyXB)
    sparse_diverse_set_continuous = sparseDiversePoolLogRegModel_object.get_sparse_diverse_set(gap_tolerance=sparseLevelSet_gap_tolerance, select_top_m=sparseLevelSet_select_top_m, maxAttempts=maxAttempts)

    sparse_diverse_set_continuous_last_5_solutions = sparse_diverse_set_continuous[:, -5:]


    # for j in range(5):
    #     nonzero_indices = np.where(np.abs(sparse_diverse_set_continuous_last_5_solutions[:, j]) > 1e-8)[0]
    #     print(nonzero_indices)
        
    #     sparse_coefficients = sparse_diverse_set_continuous_last_5_solutions[nonzero_indices, j]

    #     print(sparse_coefficients)
    
    expected_last_5_solutions = get_expected_last_5_solutions()

    assert isEqual_upTo_8decimal(expected_last_5_solutions, sparse_diverse_set_continuous_last_5_solutions), "the last 5 solutions given by sparse diverse pool algorithm is not correct!"

if __name__ == '__main__':
    test_sparseDiversePool()