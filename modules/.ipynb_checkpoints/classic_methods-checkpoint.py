from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

def cluster_classical_methods(X, n_clusters=16):
    """
    输入: X: ndarray (N, D)
         n_clusters: 聚类个数
    输出: 包含不同方法聚类结果的字典
    """
    results = {}

    # 1. KMeans
    results['KMeans'] = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)

    # 2. PCA + KMeans
    X_pca = PCA(n_components=30).fit_transform(X)
    results['PCA+KMeans'] = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X_pca)

    # 3. Spectral Clustering
    results['Spectral'] = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0).fit_predict(X)

    # 4. Agglomerative Clustering
    results['Agglomerative'] = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)

    # 5. Gaussian Mixture Model (GMM)
    results['GMM'] = GaussianMixture(n_components=n_clusters, random_state=0).fit(X).predict(X)

    return results
