import numpy as np
from fasterrisk.sparseBeamSearch import sparseLogRegModel
from fasterrisk.sparseDiversePool import sparseDiversePoolLogRegModel
from fasterrisk.rounding import starRaySearchModel
from fasterrisk.utils import isEqual_upTo_8decimal

# [ 0,  2, 10, 14, 21, 28]
# [ 0,  2, 10, 14, 21, 34]
# [ 0,  2, 10, 14, 21, 30]
# [ 0,  2, 10, 14, 21, 24]
# [ 0,  2, 10, 14, 21, 22]

# [-3, -2, -3, -5,  4,  1.]
# [-2, -2, -2, -4,  3, -1.]
# [-3, -2, -3, -5,  4,  1.]
# [-2, -2, -2, -4,  3, -1.]
# [-3, -2, -2, -4,  4,  1.]

def get_expected_last_5_multipliers():
    return np.asarray([1.79963122, 1.48820364, 1.77281803, 1.59960591, 1.60098164])

def get_expected_last_5_integer_solutions():
    expected_last_5_solutions = np.zeros((37, 5))

    expected_last_5_solutions[:, 0][np.asarray([ 0,  2, 10, 14, 21, 28], dtype=int)] = np.asarray([-3, -2, -3, -5,  4,  1.])
    expected_last_5_solutions[:, 1][np.asarray([ 0,  2, 10, 14, 21, 34], dtype=int)] = np.asarray([-2, -2, -2, -4,  3, -1.])
    expected_last_5_solutions[:, 2][np.asarray([ 0,  2, 10, 14, 21, 30], dtype=int)] = np.asarray([-3, -2, -3, -5,  4,  1.])
    expected_last_5_solutions[:, 3][np.asarray([ 0,  2, 10, 14, 21, 24], dtype=int)] = np.asarray([-2, -2, -2, -4,  3, -1.])
    expected_last_5_solutions[:, 4][np.asarray([ 0,  2, 10, 14, 21, 22], dtype=int)] = np.asarray([-3, -2, -2, -4,  4,  1.])
    return expected_last_5_solutions

def test_rounding():
    # import data
    # train_test_data = np.load("tests/train_test_data.npz", allow_pickle=True)
    train_test_data = np.load("tests/train_test_data_noIntercept.npz", allow_pickle=True)
    
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


    starRaySearchModel_object = starRaySearchModel(X = X_train, y = y_train, num_ray_search=num_ray_search, early_stop_tolerance=lineSearch_early_stop_tolerance)

    multipliers, sparse_diverse_set_integer = starRaySearchModel_object.star_ray_search_scale_and_round(sparse_diverse_set_continuous)

    multipliers_last_5, sparse_diverse_set_integer_last_5 = multipliers[-5:], sparse_diverse_set_integer[:, -5:]

    # print(multipliers_last_5)

    # for j in range(5):
    #     nonzero_indices = np.where(np.abs(sparse_diverse_set_integer_last_5[:, j]) > 1e-8)[0]
    #     print(nonzero_indices)
        
    #     sparse_coefficients = sparse_diverse_set_integer_last_5[nonzero_indices, j]

        # print(sparse_coefficients)
    
    expected_last_5_integer_solutions = get_expected_last_5_integer_solutions()


    assert isEqual_upTo_8decimal(expected_last_5_integer_solutions, sparse_diverse_set_integer_last_5), "last 5 integer coefficients are not correct!"

    expected_last_5_multipliers = get_expected_last_5_multipliers()

    assert isEqual_upTo_8decimal(expected_last_5_multipliers, multipliers_last_5), "last 5 multipliers are not correct!"

if __name__ == '__main__':
    test_rounding()