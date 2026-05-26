import numpy as np

def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or y_true.sum() == 0: return 0.0
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    prec = tp / np.arange(1, len(y_true) + 1)
    return float(np.sum(prec * y_true) / y_true.sum())

def gini_abs(ac: np.ndarray, an: np.ndarray) -> float:
    """Gini coefficient between concept (ac) and non-concept (an) activations."""
    if ac.size == 0 or an.size == 0: return 0.0
    # Comparison matrix: how often is concept > non-concept
    comp = (ac.reshape(-1, 1) > an.reshape(1, -1)).mean()
    return float(abs(2.0 * comp - 1.0))