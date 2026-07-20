from __future__ import annotations

import numpy as np

from oddd.conformal.cluster_conformal import ClusterConformalPredictor
from oddd.conformal.nonconformity import nonconformity_scores


class VanillaConformal:
    """Global split conformal without cluster conditioning."""

    def __init__(self, alpha: float = 0.1, normalized: bool = True):
        self.alpha = alpha
        self.normalized = normalized
        self.cp = ClusterConformalPredictor(alpha=alpha, min_group_size=10**9)

    def fit(self, y_cal: np.ndarray, mu_cal: np.ndarray, sigma_cal: np.ndarray | None, task_type: str):
        scores = nonconformity_scores(
            y_cal, mu_cal, task_type, sigma=sigma_cal, normalized=self.normalized
        )
        groups = np.zeros(len(scores), dtype=np.int64)
        self.cp.fit(scores, groups)
        return self

    def predict(self, mu: np.ndarray, task_type: str):
        groups = np.zeros(len(mu), dtype=np.int64)
        return self.cp.predict_interval(mu, groups, task_type)
