import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")
from fasterrisk.utils import get_support_indices, get_nonsupport_indices, normalize_X, compute_logisticLoss_from_yXB, compute_logisticLoss_from_ExpyXB, compute_logisticLoss_from_X_y_beta0_betas
from fasterrisk.base_model import logRegModel
   
class sparseLogRegModel(logRegModel):
    def __init__(self, X, y, lambda2=1e-8, intercept=True, original_lb=-5, original_ub=5):
        super().__init__(X=X, y=y, lambda2=lambda2, intercept=intercept, original_lb=original_lb, original_ub=original_ub)
   
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
