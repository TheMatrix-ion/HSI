import scipy.io as sio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

def load_hsi_data(mat_file_path, key='salinas_corrected'):
    """读取 ``mat_file_path`` 中的 HSI 数据

    Parameters
    ----------
    mat_file_path : str
        ``.mat`` 文件路径
    key : str, optional
        希望从 ``.mat`` 中加载的键名。当提供的 ``key``
        在文件中存在时将直接返回对应数据。

    Returns
    -------
    numpy.ndarray
        加载的 HSI 数据，``dtype`` 为 ``float32``，形状为 ``(H, W, C)``
    """
    mat = sio.loadmat(mat_file_path)

    # 如果指定的 key 存在，优先返回对应的数据
    if key and key in mat:
        data = mat[key]
    else:
        # 回退: 取第一个非内部字段
        data = None
        for k in mat:
            if not k.startswith("__"):
                data = mat[k]
                break
        if data is None:
            raise KeyError(f"No valid data key found in {mat_file_path}")

    return data.astype(np.float32)  # shape: (H, W, C)

def normalize(X):
    """
    将每个像素的光谱归一化到 [0, 1]
    输入形状: (H, W, C) 或 (N, C)
    """
    original_shape = X.shape
    if len(X.shape) == 3:
        H, W, C = X.shape
        X = X.reshape(-1, C)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    if len(original_shape) == 3:
        return X_scaled.reshape(H, W, C)
    else:
        return X_scaled

def visualize_clusters(pred_labels, H, W, title='Cluster Map', save_path='views/cluster_result.png'):
    pred_map = pred_labels.reshape(H, W)
    plt.figure(figsize=(5, 5))
    plt.imshow(pred_map, cmap='tab20', interpolation='none')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster result to {save_path}")

def load_ground_truth(gt_path):
    mat = sio.loadmat(gt_path)
    for k in mat:
        if not k.startswith('__'):
            return mat[k].astype(int)
    raise ValueError("No valid GT key found in .mat file")

def visualize_ground_truth(gt_array, title='Ground Truth', save_path='views/ground_truth.png'):
    plt.figure(figsize=(5, 5))
    plt.imshow(gt_array, cmap='tab20', interpolation='none')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ground truth image to {save_path}")

def plot_comparison(gt_array, pred_labels, H, W, save_path='views/comparison.png'):
    pred_map = pred_labels.reshape(H, W)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt_array, cmap='tab20', interpolation='none')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    axes[1].imshow(pred_map, cmap='tab20', interpolation='none')
    axes[1].set_title('Transformer + KMeans')
    axes[1].axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison image to {save_path}")


