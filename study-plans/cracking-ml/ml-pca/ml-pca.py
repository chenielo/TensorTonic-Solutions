import numpy as np

def pca(X: list, n_components: int = 2) -> tuple:
    """
    Returns transformed rows and explained variance ratios.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    X_mean = np.mean(X, axis=0)
    Xc = X-X_mean 
    cov = (1/(n-1))*Xc.T@Xc
    eigenvalues,eigenvectors=np.linalg.eigh(cov)
    order = np.argsort(-eigenvalues,kind="stable")
    eigenvalues=eigenvalues[order]
    eigenvectors=eigenvectors[:,order]
    for c in range(eigenvectors.shape[1]):
        pivot = int(np.argmax(np.abs(eigenvectors[:, c])))
        if eigenvectors[pivot,c]<0:
            eigenvectors[:,c]*=-1
    components=eigenvectors[:,:n_components]
    transformed = Xc@components
    tot =float(np.sum(eigenvalues))
    ratios = eigenvalues[:n_components]/tot 
    return [[[round(float(v),4) for v in row] for row in transformed],
    [round(float(v),4) for v in ratios]
    ]
    
