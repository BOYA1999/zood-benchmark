from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConformalResult:
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    group_ids: np.ndarray
    quantiles: dict[int, float] = field(default_factory=dict)


class ClusterConformalPredictor:
    """Cluster-conditioned split conformal prediction with Mondrian-style groups."""

    def __init__(
        self,
        alpha: float = 0.1,
        min_group_size: int = 15,
        conservative_multiplier: float = 1.5,
    ):
        self.alpha = alpha
        self.min_group_size = min_group_size
        self.conservative_multiplier = conservative_multiplier
        self.quantiles_: dict[int, float] = {}
        self.global_quantile_: float = 0.0
        self.ood_group_id_: int = -1

    @staticmethod
    def _finite_quantile(scores: np.ndarray, alpha: float) -> float:
        if len(scores) == 0:
            return 0.0
        q_level = np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores)
        q_level = min(max(q_level, 0.0), 1.0)
        return float(np.quantile(scores, q_level, method="higher"))

    def fit(self, scores: np.ndarray, groups: np.ndarray) -> "ClusterConformalPredictor":
        self.global_quantile_ = self._finite_quantile(scores, self.alpha)
        self.quantiles_.clear()
        unique = np.unique(groups)
        for g in unique:
            mask = groups == g
            if mask.sum() < self.min_group_size:
                continue
            self.quantiles_[int(g)] = self._finite_quantile(scores[mask], self.alpha)
        self.ood_group_id_ = int(unique.max()) + 1 if len(unique) else 0
        ood_q = self.global_quantile_ * self.conservative_multiplier
        self.quantiles_[self.ood_group_id_] = ood_q
        return self

    def _group_quantile(self, group_ids: np.ndarray) -> np.ndarray:
        q = np.full(len(group_ids), self.global_quantile_, dtype=np.float64)
        for i, g in enumerate(group_ids):
            q[i] = self.quantiles_.get(int(g), self.quantiles_.get(self.ood_group_id_, self.global_quantile_))
        return q

    def predict_interval(
        self,
        y_pred: np.ndarray,
        group_ids: np.ndarray,
        task_type: str,
    ) -> ConformalResult:
        q = self._group_quantile(group_ids)
        if task_type == "regression":
            lower = y_pred - q
            upper = y_pred + q
        else:
            lower = np.clip(y_pred - q, 0.0, 1.0)
            upper = np.clip(y_pred + q, 0.0, 1.0)
        width = upper - lower
        return ConformalResult(
            lower=lower,
            upper=upper,
            width=width,
            group_ids=group_ids,
            quantiles={int(k): float(v) for k, v in self.quantiles_.items()},
        )
