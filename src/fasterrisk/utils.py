import numpy as np

def get_support_indices(betas):
    return np.where(np.abs(betas) > 1e-9)[0]

def get_nonsupport_indices(betas):
    return np.where(np.abs(betas) <= 1e-9)[0]

def normalize_X(X):
    X_mean = np.mean(X, axis=0)
    X_norm = np.linalg.norm(X-X_mean, axis=0)
    scaled_feature_indices = np.where(X_norm >= 1e-9)[0]
    X_normalized = X-X_mean
    X_normalized[:, scaled_feature_indices] = X_normalized[:, scaled_feature_indices]/X_norm[[scaled_feature_indices]]
    return X_normalized, X_mean, X_norm, scaled_feature_indices

def compute_logisticLoss_from_yXB(yXB):
    # shape of yXB is (n, )
    return np.sum(np.log(1.+np.exp(-yXB)))

def compute_logisticLoss_from_ExpyXB(ExpyXB):
    # shape of ExpyXB is (n, )
    return np.sum(np.log(1.+np.reciprocal(ExpyXB)))

def compute_logisticLoss_from_betas_and_yX(betas, yX):
    # shape of betas is (p, )
    # shape of yX is (n, p)
    yXB = yX.dot(betas)
    return compute_logisticLoss_from_yXB(yXB)

def compute_logisticLoss_from_X_y_beta0_betas(X, y, beta0, betas):
    XB = X.dot(betas) + beta0
    yXB = y * XB
    return compute_logisticLoss_from_yXB(yXB)

def convert_y_to_neg_and_pos_1(y):
    y_max, y_min = np.min(y), np.max(y)
    y_transformed = -1 + 2 * (y-y_min)/(y_max-y_min) # convert y to -1 and 1
    return y_transformed

# def get_acc_and_auc(betas, X, y):
#     accuracy = compute_accuracy(betas, X, y)
#     auc = compute_auc(betas, X, y)
#     return accuracy, auc

# def compute_accuracy(betas, X, y):
#     y_pred = X.dot(betas)
#     y_pred = 2 * (y_pred > 0) - 1
#     return np.sum(y_pred == y) / y.shape[0]

# def compute_auc(betas, X, y):
#     y_pred = X.dot(betas)
#     y_pred = 1/(1+np.exp(-y_pred))
#     fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_pred)
#     auc = sklearn.metrics.auc(fpr, tpr)
#     return auc

def isEqual_upTo_8decimal(a, b):
    if np.isscalar(a):
        return abs(a - b) < 1e-8
    return np.max(np.abs(a - b)) < 1e-8

def isEqual_upTo_16decimal(a, b):
    if np.isscalar(a):
        return abs(a - b) < 1e-16
    return np.max(np.abs(a - b)) < 1e-16

def insertIntercept_asFirstColOf_X(X):
    n = len(X)
    intercept = np.ones((n, 1))
    X_with_intercept = np.hstack((intercept, X))
    return X_with_intercept