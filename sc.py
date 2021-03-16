from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans, SpectralClustering

def spectral_clustering(S, param=1e-2):
    laplacian_norm = laplacian(S, normed=True)
    eigen_values, eigen_vectors = np.linalg.eig(laplacian_norm)
    eigen_values = np.real(eigen_values)
    eigen_vectors = np.real(eigen_vectors)
    
    P = eigen_vectors.T[eigen_values < param]

    km = KMeans(n_clusters=P.shape[0])  
    return km.fit_predict(P.T)
