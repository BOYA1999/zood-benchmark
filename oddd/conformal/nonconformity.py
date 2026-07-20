from __future__ import annotations

import numpy as np


def nonconformity_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    sigma: np.ndarray | None = None,
    normalized: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute nonconformity scores for regression or classification."""
    if task_type == "regression":
        scores = np.abs(y_true - y_pred)
        if normalized and sigma is not None:
            scores = scores / (sigma + eps)
        return scores.astype(np.float64)

    # classification: 1 - predicted probability of true class (not variance-normalized)
    y_true = y_true.astype(int)
    p = np.clip(y_pred, eps, 1.0 - eps)
    scores = np.where(y_true == 1, 1.0 - p, p)
    return scores.astype(np.float64)
