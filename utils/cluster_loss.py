import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_assign(z, prototypes, temperature=1.0):
    """
    使用 Student's t 分布生成对每个原型的 soft assignment 概率
    z: (N, D) 样本嵌入
    prototypes: (K, D) 原型向量
    return: q (N, K)
    """
    dist = torch.cdist(z, prototypes)  # 欧式距离
    q = (1.0 + dist**2 / temperature)**-1
    q = q / torch.sum(q, dim=1, keepdim=True)  # softmax over clusters
    return q

def kl_cluster_loss(q):
    """
    KL divergence between soft assignment q and its target distribution p
    """
    f = torch.sum(q, dim=0)
    p = (q**2) / f
    p = p / torch.sum(p, dim=1, keepdim=True)
    loss = F.kl_div(q.log(), p, reduction='batchmean')
    return loss
