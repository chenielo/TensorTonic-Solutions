import numpy as np

def feature_scale(X: list, method: str = "minmax") -> list:
    """
    Returns the independently scaled feature columns.
    """
    values = np.asarray(X)
    if method == "minmax":
        xmin = np.min(values,axis=0)
        xmax = np.max(values,axis=0)
        denominator=xmax-xmin
        denominator=np.where(denominator==0.0,1.0,denominator)
        x = (values-xmin)/denominator
    else:
        xmean = np.mean(values,axis=0)
        xstd = np.std(values,axis=0)
        denominator = xstd
        denominator=np.where(denominator==0.0,1.0,denominator)

        x = (values - xmean)/denominator 
    return x 
        
