import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    if len(docs) == 0:
        return np.zeros(0, dtype=float)
    lengths = np.array([len(d) for d in docs], dtype=float)
    avg_len = float(np.mean(lengths))
    freq = [Counter(d) for d in docs]
    doc_freq = Counter()
    for d in docs:
        doc_freq.update(set(d))
    scores = np.zeros(len(docs),dtype=float)
    for term in dict.fromkeys(query_tokens):
        f = doc_freq[term]
        if f == 0: 
            continue
        idf = math.log((len(docs)-f+0.5)/(f+0.5)+1.0)
        term_freq = np.array([counts[term] for counts in freq])
        length_factor = 1.0 -b + b*lengths/avg_len
        denom = term_freq + k1*length_factor
        scores += idf * term_freq *(k1+1.0) / denom
    return scores