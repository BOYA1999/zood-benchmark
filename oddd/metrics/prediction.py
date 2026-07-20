from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    out: dict[str, float | None] = {"auroc": None, "auprc": None, "n_test": int(len(y_true)), "positive_rate": float(y_true.mean())}
    if len(np.unique(y_true)) < 2:
        return out
    out["auroc"] = float(roc_auc_score(y_true, y_prob))
    out["auprc"] = float(average_precision_score(y_true, y_prob))
    return out
