import numpy as np

def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or y_true.sum() == 0: return 0.0
    order = np.argsort(-y_score, kind="stable")
    y_score = y_score[order]
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    # A tied score defines one threshold, regardless of input row order.
    ends = np.r_[np.flatnonzero(np.diff(y_score)), len(y_score) - 1]
    positives = np.diff(np.r_[0, tp[ends]])
    return float(np.sum(positives * tp[ends] / (ends + 1)) / y_true.sum())

def gini_abs(ac: np.ndarray, an: np.ndarray) -> float:
    """Gini coefficient between concept (ac) and non-concept (an) activations."""
    if ac.size == 0 or an.size == 0: return 0.0
    # Comparison matrix: how often is concept > non-concept
    # Ties contribute half a win; identical inactive populations score zero.
    # Sorted lookup also avoids allocating a synthetic-by-natural matrix.
    ordered = np.sort(an)
    lower = np.searchsorted(ordered, ac, side="left")
    upper = np.searchsorted(ordered, ac, side="right")
    comp = np.mean((lower + upper) / (2.0 * an.size))
    return float(abs(2.0 * comp - 1.0))
