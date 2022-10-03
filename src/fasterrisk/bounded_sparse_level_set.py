import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")
from fasterrisk.utils import get_support_indices, get_nonsupport_indices, normalize_X, compute_logisticLoss_from_yXB, compute_logisticLoss_from_ExpyXB, compute_logisticLoss_from_X_y_beta0_betas


class sparseLogRegModel:
    def __init__(self, X, y, lambda2=1e-8, intercept=True, original_lb=-5, original_ub=5):
        """
        Parameters
        ----------
        self.original_beta0:                scalar for the intercept used for output of the class
        self.original_betas:                (p, ) for the coefficients used for output
        self.X:                             (n, p) feature matrix with n sampels and p features
        self.n, self.p:                     number of samples and features in self.X
        self.X_normalized:                  (p, p) normalized feature matrix, each column has mean=0 and l2 norm=1
        self.X_mean:                        (p, ) storing the column mean of self.X
        self.X_norm:                        (p, ) storing the column l2 norm of (self.X - self.X_mean)
        self.X_scaled_feature_indices       an array storing the column indices of which column (self.X - self.X_mean) has norm 0
        self.y:                             (n, ) storing the binary classification label of {-1, 1}
        self.beta0                          scalar for the intercept used in the internal calculation of the class
        self.betas:                         (p, ) for the coefficients used in the internal calculation of the class
        self.yX:                            (n, p) self.yX[i, j] = self.y[i] * self.X_normalized[i, j]
        self.ExpyXB:                        (n, ) self.ExpyXB[i] = Exp(self.y[i] * (self.b0 + (sum_{j=1}^p self.X[i, j]*self.betas[j])))
        self.Lipschitz                      Lipschitz constant, for self.X_normalized, self.Lipschitz=1/4
        self.intercept                      flag(T/F) denotes whether there is an intercept in the model
        self.lambda2                        lambda2 regularization in the normalized feature space, i.e., self.X
        self.lbs                            (p, ) lower bounds of each coefficient self.betas
        self.ubs                            (p, ) upper bounds of each coefficient self.betas
        """
        # 
        if intercept is True:
            self.X = X[:, 1:] # (n, p)
        else:
            print("need to provide an intercept!")
            sys.exit()
        self.X_normalized, self.X_mean, self.X_norm, self.scaled_feature_indices = normalize_X(self.X)
        self.n, self.p = self.X_normalized.shape
        self.y = y.reshape(-1).astype(float)
        self.yX = y.reshape(-1, 1) * self.X_normalized
        # self.yXT = np.transpose(self.yX)
        self.yXT = np.zeros((self.p, self.n))
        self.yXT[:] = np.transpose(self.yX)[:]
        self.beta0 = 0
        self.betas = np.zeros((self.p, ))
        self.ExpyXB = np.exp(self.y * self.beta0 + self.yX.dot(self.betas))

        self.intercept = intercept
        self.lambda2 = lambda2
        self.twoLambda2 = 2 * self.lambda2

        self.Lipschitz = 0.25 + self.twoLambda2
        self.lbs = original_lb * np.ones(self.p)
        self.lbs[self.scaled_feature_indices] *= self.X_norm[self.scaled_feature_indices]
        self.ubs = original_ub * np.ones(self.p)
        self.ubs[self.scaled_feature_indices] *= self.X_norm[self.scaled_feature_indices]
    
    def get_warm_start(self, betas_initial):
        # betas_initial has dimension (p+1, 1)
        self.original_beta0 = betas_initial[0][0]
        self.original_betas = betas_initial[1:].reshape(-1)
        self.beta0, self.betas = self.transform_coefficients_to_normalized_space(self.original_beta0, self.original_betas)
        print("warmstart solution in normalized space is {} and {}".format(self.beta0, self.betas))
        self.ExpyXB = np.exp(self.y * self.beta0 + self.yX.dot(self.betas))

    def expand_parent_i_support_via_OMP_by_1(self, i, sub_beam_size=20):

        non_support = get_nonsupport_indices(self.betas_arr_parent[i])
        support = get_support_indices(self.betas_arr_parent[i])


        grad_on_non_support = self.yXT[non_support].dot(np.reciprocal(1+self.ExpyXB_arr_parent[i]))
        abs_grad_on_non_support = np.abs(grad_on_non_support)

        num_new_js = min(sub_beam_size, len(non_support))
        new_js = non_support[np.argsort(-abs_grad_on_non_support)][:num_new_js]
        child_start, child_end = i*sub_beam_size, i*sub_beam_size + num_new_js

        self.ExpyXB_arr_child[child_start:child_end] = self.ExpyXB_arr_parent[i, :] # (num_new_js, n)
        self.betas_arr_child[child_start:child_end, non_support] = 0
        self.betas_arr_child[child_start:child_end, support] = self.betas_arr_parent[i, support]
        self.beta0_arr_child[child_start:child_end] = self.beta0_arr_parent[i]
        
        beta_new_js = np.zeros((num_new_js, )) #(len(new_js), )
        diff_max = 1e3

        step = 0
        while step < 10 and diff_max > 1e-3:
            prev_beta_new_js = beta_new_js.copy()
            grad_on_new_js = -np.sum(self.yXT[new_js] * np.reciprocal(1.+self.ExpyXB_arr_child[child_start:child_end]), axis=1) + self.twoLambda2 * beta_new_js
            step_at_new_js = grad_on_new_js / self.Lipschitz

            beta_new_js = prev_beta_new_js - step_at_new_js
            beta_new_js = np.clip(beta_new_js, self.lbs[new_js], self.ubs[new_js])
            diff_beta_new_js = beta_new_js - prev_beta_new_js

            self.ExpyXB_arr_child[child_start:child_end] *= np.exp(self.yXT[new_js] * diff_beta_new_js.reshape(-1, 1))

            diff_max = max(np.abs(diff_beta_new_js))
            step += 1

        for l in range(num_new_js):
            child_id = child_start + l
            self.betas_arr_child[child_id, new_js[l]] = beta_new_js[l]
            tmp_support_str = str(get_support_indices(self.betas_arr_child[child_id]))
            if tmp_support_str not in self.forbidden_support:
                self.forbidden_support.add(tmp_support_str)

                self.ExpyXB_arr_child[child_id], self.beta0_arr_child[child_id], self.betas_arr_child[child_id] = self.finetune_on_current_support(self.ExpyXB_arr_child[child_id], self.beta0_arr_child[child_id], self.betas_arr_child[child_id])
                self.loss_arr_child[child_id] = compute_logisticLoss_from_ExpyXB(self.ExpyXB_arr_child[child_id])

    def beamSearch_multipleSupports_via_OMP_by_1(self, beam_size=10, sub_beam_size=20):
        self.loss_arr_child.fill(1e12)

        for i in range(self.num_parent):
            self.expand_parent_i_support_via_OMP_by_1(i, sub_beam_size=sub_beam_size)

        child_indices = np.argsort(self.loss_arr_child)[:beam_size] # get indices of children which have the smallest losses
        num_child_indices = len(child_indices)
        self.ExpyXB_arr_parent[:num_child_indices], self.beta0_arr_parent[:num_child_indices], self.betas_arr_parent[:num_child_indices] = self.ExpyXB_arr_child[child_indices], self.beta0_arr_child[child_indices], self.betas_arr_child[child_indices]

        self.num_parent = num_child_indices

    def get_sparse_sol_via_OMP(self, k=0, beam_size=10, sub_beam_size=20):
        # get a sparse solution with specificed sparsity level=5
        # through orthogonal matching pursuit method
        nonzero_indices_set = set(np.where(np.abs(self.betas) > 1e-9)[0])
        print("get_sparse_sol_via_OMP, initial support is:", nonzero_indices_set)
        zero_indices_set = set(range(self.p)) - nonzero_indices_set
        num_nonzero = len(nonzero_indices_set)

        if len(zero_indices_set) == 0:
            return

        # if there is no warm start solution, initialize beta0 analytically
        if len(nonzero_indices_set) == 0:
            y_sum = np.sum(self.y)
            num_y_pos_1 = (y_sum + self.n)/2
            num_y_neg_1 = self.n - num_y_pos_1
            self.beta0 = np.log(num_y_pos_1/num_y_neg_1)
            self.ExpyXB *= np.exp(self.y * self.beta0)

        # create beam search parent
        self.ExpyXB_arr_parent = np.zeros((beam_size, self.n))
        self.beta0_arr_parent = np.zeros((beam_size, ))
        self.betas_arr_parent = np.zeros((beam_size, self.p))
        self.ExpyXB_arr_parent[0, :] = self.ExpyXB[:]
        self.beta0_arr_parent[0] = self.beta0
        self.betas_arr_parent[0, :] = self.betas[:]
        self.num_parent = 1

        # create beam search children. parent[i]->child[i*sub_beam_size:(i+1)*sub_beam_size]
        total_sub_beam_size = beam_size * sub_beam_size
        self.ExpyXB_arr_child = np.zeros((total_sub_beam_size, self.n))
        self.beta0_arr_child = np.zeros((total_sub_beam_size, ))
        self.betas_arr_child = np.zeros((total_sub_beam_size, self.p))
        self.isMasked_arr_child = np.ones((total_sub_beam_size, ), dtype=bool)
        self.loss_arr_child = 1e12 * np.ones((total_sub_beam_size, ))
        self.forbidden_support = set()

        while num_nonzero < min(k, self.p):
            num_nonzero += 1

            self.beamSearch_multipleSupports_via_OMP_by_1(beam_size=beam_size, sub_beam_size=sub_beam_size)

        # then optimize on the new expanded support
        self.ExpyXB, self.beta0, self.betas = self.ExpyXB_arr_parent[0], self.beta0_arr_parent[0], self.betas_arr_parent[0]
    
    def get_original_beta0_and_betas(self):
        return self.transform_coefficients_to_original_space(self.beta0, self.betas)
    
    def transform_coefficients_to_original_space(self, beta0, betas):
        original_betas = betas.copy()
        original_betas[self.scaled_feature_indices] = original_betas[self.scaled_feature_indices]/self.X_norm[self.scaled_feature_indices]
        original_beta0 = beta0 - np.dot(self.X_mean, original_betas)
        return original_beta0, original_betas

    def transform_coefficients_to_normalized_space(self, original_beta0, original_betas):
        betas = original_betas.copy()
        betas[self.scaled_feature_indices] = betas[self.scaled_feature_indices] * self.X_norm[self.scaled_feature_indices]
        beta0 = original_beta0 + self.X_mean.dot(original_betas)
        return beta0, betas

    def get_grad_at_coord(self, ExpyXB, betas_j, yX_j, j):
        # return -np.dot(1/(1+ExpyXB), self.yX[:, j]) + self.twoLambda2 * betas_j
        # return -np.inner(1/(1+ExpyXB), self.yX[:, j]) + self.twoLambda2 * betas_j
        # return -np.inner(np.reciprocal(1+ExpyXB), self.yX[:, j]) + self.twoLambda2 * betas_j
        return -np.inner(np.reciprocal(1+ExpyXB), yX_j) + self.twoLambda2 * betas_j
        # return -yX_j.dot(np.reciprocal(1+ExpyXB)) + self.twoLambda2 * betas_j

    def update_ExpyXB(self, ExpyXB, yX_j, diff_betas_j):
        ExpyXB *= np.exp(yX_j * diff_betas_j)
    
    def optimize_1step_at_coord(self, ExpyXB, betas, yX_j, j):
        # in-place modification, heck that ExpyXB and betas are passed by reference
        prev_betas_j = betas[j]
        current_betas_j = prev_betas_j
        grad_at_j = self.get_grad_at_coord(ExpyXB, current_betas_j, yX_j, j)
        step_at_j = grad_at_j / self.Lipschitz
        current_betas_j = prev_betas_j - step_at_j
        # current_betas_j = np.clip(current_betas_j, self.lbs[j], self.ubs[j])
        current_betas_j = max(self.lbs[j], min(self.ubs[j], current_betas_j))
        diff_betas_j = current_betas_j - prev_betas_j
        betas[j] = current_betas_j

        # ExpyXB *= np.exp(yX_j * diff_betas_j)
        self.update_ExpyXB(ExpyXB, yX_j, diff_betas_j)

    def finetune_on_current_support(self, ExpyXB, beta0, betas, total_CD_steps=100):

        support  = np.where(np.abs(betas) > 1e-9)[0]
        grad_on_support = -self.yXT[support].dot(np.reciprocal(1+ExpyXB)) + self.twoLambda2 * betas[support]
        abs_grad_on_support = np.abs(grad_on_support)
        support = support[np.argsort(-abs_grad_on_support)]

        loss_before = compute_logisticLoss_from_ExpyXB(ExpyXB) + self.lambda2 * betas[support].dot(betas[support])
        for steps in range(total_CD_steps): # number of iterations for coordinate descent

            if self.intercept:
                grad_intercept = -np.reciprocal(1+ExpyXB).dot(self.y)
                step_at_intercept = grad_intercept / (self.n * 0.25) # lipschitz constant is 0.25 at the intercept
                beta0 = beta0 - step_at_intercept
                ExpyXB *= np.exp(self.y * (-step_at_intercept))

            for j in support:
                self.optimize_1step_at_coord(ExpyXB, betas, self.yXT[j, :], j) # in-place modification on ExpyXB and betas
            
            if steps % 10 == 0:
                loss_after = compute_logisticLoss_from_ExpyXB(ExpyXB) + self.lambda2 * betas[support].dot(betas[support])
                if abs(loss_before - loss_after)/loss_after < 1e-8:
                    # print("break after {} steps; support size is {}".format(steps, len(support)))
                    break
                loss_before = loss_after
        
        return ExpyXB, beta0, betas
    
    def compute_yXB(self, beta0, betas):
        return self.y*(beta0 + np.dot(self.X_normalized, betas))
    
    def get_sparse_diverse_set(self, gap_tolerance=0.05, select_top_m=10, maxAttempts=50):
        # select top m solutions with the lowest logistic losses
        # Note Bene: loss comparison here does not include logistic loss
        nonzero_indices = get_support_indices(self.betas)
        zero_indices = get_nonsupport_indices(self.betas)

        num_support = len(nonzero_indices)
        num_nonsupport = len(zero_indices)

        maxAttempts = min(maxAttempts, num_nonsupport)
        max_num_new_js = maxAttempts

        total_solutions = 1 + num_support * maxAttempts
        sparse_level_set_betas = np.zeros((total_solutions, self.p))
        sparse_level_set_betas[:, nonzero_indices] = self.betas[nonzero_indices]

        sparse_level_set_beta0 = self.beta0 * np.ones((total_solutions, ))
        sparse_level_set_ExpyXB = np.zeros((total_solutions, self.n))
        sparse_level_set_loss = 1e12 * np.ones((total_solutions, ))

        sparse_level_set_ExpyXB[-1] = self.ExpyXB
        sparse_level_set_loss[-1] = compute_logisticLoss_from_ExpyXB(self.ExpyXB) + self.lambda2 * self.betas[nonzero_indices].dot(self.betas[nonzero_indices])

        # nonzero_indices_set = set(nonzero_indices)

        betas_squareSum = self.betas[nonzero_indices].dot(self.betas[nonzero_indices])
        for num_old_j, old_j in enumerate(nonzero_indices):
            # pick $maxAttempt$ number of features that can replace old_j
            sparse_level_start = num_old_j * maxAttempts
            sparse_level_end = (1 + num_old_j) * maxAttempts

            sparse_level_set_ExpyXB[sparse_level_start:sparse_level_end] = self.ExpyXB * np.exp(-self.yXT[old_j] * self.betas[old_j])

            sparse_level_set_betas[sparse_level_start:sparse_level_end, old_j] = 0
            
            betas_no_old_j_squareSum = betas_squareSum - self.betas[old_j]**2

            grad_on_nonsupport = -self.yXT[zero_indices].dot(np.reciprocal(1+sparse_level_set_ExpyXB[sparse_level_start]))
            abs_grad_on_nonsupport = np.abs(grad_on_nonsupport)

            # new_js = np.argpartition(abs_full_grad, -max_num_new_js)[-max_num_new_js:]
            new_js = zero_indices[np.argsort(-abs_grad_on_nonsupport)[:max_num_new_js]]

            for num_new_j, new_j in enumerate(new_js):
                sparse_level_set_index = sparse_level_start + num_new_j
                for _ in range(10):
                    self.optimize_1step_at_coord(sparse_level_set_ExpyXB[sparse_level_set_index], sparse_level_set_betas[sparse_level_set_index], self.yXT[new_j, :], new_j)
                
                loss_sparse_level_set_index = compute_logisticLoss_from_ExpyXB(sparse_level_set_ExpyXB[sparse_level_set_index]) + self.lambda2 * (betas_no_old_j_squareSum + sparse_level_set_betas[sparse_level_set_index, new_j] ** 2)

                if (loss_sparse_level_set_index - sparse_level_set_loss[-1]) / sparse_level_set_loss[-1] < gap_tolerance:
                    sparse_level_set_ExpyXB[sparse_level_set_index], sparse_level_set_beta0[sparse_level_set_index], sparse_level_set_betas[sparse_level_set_index] = self.finetune_on_current_support(sparse_level_set_ExpyXB[sparse_level_set_index], sparse_level_set_beta0[sparse_level_set_index], sparse_level_set_betas[sparse_level_set_index])

                    sparse_level_set_loss[sparse_level_set_index] = compute_logisticLoss_from_ExpyXB(sparse_level_set_ExpyXB[sparse_level_set_index]) + self.lambda2 * (betas_no_old_j_squareSum + sparse_level_set_betas[sparse_level_set_index, new_j] ** 2)

        selected_sparse_level_set_indices = np.argsort(sparse_level_set_loss)[:select_top_m]

        tmp_betas = sparse_level_set_betas[selected_sparse_level_set_indices]
        tmp_beta0 = sparse_level_set_beta0[selected_sparse_level_set_indices]

        original_sparse_level_set_solution = np.zeros((1 + self.p, select_top_m))

        original_sparse_level_set_solution[1:] = sparse_level_set_betas[selected_sparse_level_set_indices].T
        original_sparse_level_set_solution[1+self.scaled_feature_indices] /= self.X_norm[self.scaled_feature_indices].reshape(-1, 1)

        original_sparse_level_set_solution[0] = sparse_level_set_beta0[selected_sparse_level_set_indices]
        original_sparse_level_set_solution[0] -= self.X_mean.T @ original_sparse_level_set_solution[1:]
        
        return original_sparse_level_set_solution # (1+p, m) m is the number of solutions in the level set