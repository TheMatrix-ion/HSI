import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.cluster import KMeans
from modules.autoencoder import Autoencoder
from modules.transformer_model import TransformerClusterNet
from modules.classic_methods import cluster_classical_methods
from utils.preprocessing import load_hsi_data, normalize, visualize_clusters, load_ground_truth, plot_comparison
from utils.evaluation import evaluate_all

# 设置
DATA_PATH = 'data/Salinas_corrected.mat'
GT_PATH = 'data/Salinas_gt.mat'
N_CLUSTERS = 16   # Salinas 有 16 类
EPOCHS = 50
LATENT_DIM = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # 1. 加载并预处理数据
    print("Loading data...")
    X = load_hsi_data(DATA_PATH)  # (H, W, C)
    H, W, C = X.shape
    X = normalize(X)
    X = X.reshape(-1, C)  # (N, C)

    # 2. 构建模型
    print("Building models...")
    autoencoder = Autoencoder(input_dim=X.shape[1], latent_dim=LATENT_DIM).to(DEVICE)
    transformer = TransformerClusterNet(embed_dim=LATENT_DIM, output_dim=LATENT_DIM).to(DEVICE)

    # 3. 训练 Autoencoder
    print("Training autoencoder...")
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    autoencoder.train()
    for epoch in range(EPOCHS):
        x_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        _, x_recon = autoencoder(x_tensor)
        loss = loss_fn(x_recon, x_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:02d}: MSE Loss = {loss.item():.4f}")

    # 4. Transformer 特征提取
    print("Extracting features with transformer...")
    autoencoder.eval()
    with torch.no_grad():
        features, _ = autoencoder(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        embed = transformer(features).cpu().numpy()

    # 5. 聚类（Transformer + KMeans）
    print("Clustering with KMeans...")
    pred_labels = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit_predict(embed)

    # 6. 可视化 + 评估
    visualize_clusters(pred_labels, H, W, title='Transformer + KMeans Clustering', save_path='views/transformer_kmeans.png')
    gt = load_ground_truth(GT_PATH).reshape(-1)
    mask = gt > 0
    scores = evaluate_all(gt[mask], pred_labels[mask])
    print("[Transformer + KMeans] Evaluation:", scores)

    # 7. 对比其他方法
    # print("Evaluating classical clustering methods...")
    # results = cluster_classical_methods(X, n_clusters=N_CLUSTERS)
    # for method, labels in results.items():
    #     masked_labels = labels[mask]
    #     metrics = evaluate_all(gt[mask], masked_labels)
    #     print(f"[{method}] Evaluation:", metrics)

if __name__ == '__main__':
    main()
