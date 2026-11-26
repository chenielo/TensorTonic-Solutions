import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    pos = np.arange(seq_length, dtype=float)[:, None] 
    i = np.arange(0, d_model, 2, dtype=float)
    angle = pos/np.power(10000.0, i/d_model)
    pe = np.zeros((seq_length, d_model), dtype=float)
    pe[:,0::2] = np.sin(angle)
    pe[:,1::2] = np.cos(angle)
    return pe
    