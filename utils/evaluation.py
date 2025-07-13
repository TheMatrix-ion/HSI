import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.optimize import linear_sum_assignment
from collections import Counter

def best_map(true_labels, pred_labels):
    """
    对聚类标签进行最佳匹配（Hungarian算法），使得聚类标签尽可能接近真实标签
    """
    true_labels = true_labels.astype(np.int64)
    pred_labels = pred_labels.astype(np.int64)
    D = max(pred_labels.max(), true_labels.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(len(pred_labels)):
        cost_matrix[pred_labels[i], true_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    new_pred = np.zeros_like(pred_labels)
    # row_ind/col_ind may be shorter than D, so only iterate over the
    # returned Hungarian assignments to avoid index errors
    for i in range(len(row_ind)):
        new_pred[pred_labels == row_ind[i]] = col_ind[i]
    return new_pred

def evaluate_all(true_labels, pred_labels):
    pred_aligned = best_map(true_labels, pred_labels)
    oa = accuracy_score(true_labels, pred_aligned)
    kappa = cohen_kappa_score(true_labels, pred_aligned)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return {'OA': oa, 'Kappa': kappa, 'NMI': nmi, 'ARI': ari}
