from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = y_true.astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def coverage_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: np.ndarray | None = None,
    alpha: float = 0.1,
) -> dict[str, float]:
    covered = (y_true >= lower) & (y_true <= upper)
    out = {
        "marginal_coverage": float(covered.mean()),
        "nominal_coverage": float(1.0 - alpha),
        "coverage_gap": float(abs(covered.mean() - (1.0 - alpha))),
    }
    if groups is not None:
        group_cov = []
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() == 0:
                continue
            group_cov.append(covered[mask].mean())
        if group_cov:
            out["worst_group_coverage"] = float(min(group_cov))
            out["mean_group_coverage"] = float(np.mean(group_cov))
    return out


def interval_width_stats(width: np.ndarray, groups: np.ndarray | None = None) -> dict[str, float]:
    out = {
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
    }
    if groups is not None:
        per_group = []
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum():
                per_group.append(width[mask].mean())
        if per_group:
            out["worst_group_mean_width"] = float(max(per_group))
    return out
