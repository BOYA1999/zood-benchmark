from __future__ import annotations

import numpy as np


def enrichment_factor(y_true: np.ndarray, scores: np.ndarray, top_frac: float) -> float:
    y_true = y_true.astype(int)
    n = len(y_true)
    k = max(1, int(n * top_frac))
    order = np.argsort(-scores)
    top_hits = y_true[order[:k]].sum()
    base_rate = y_true.mean()
    if base_rate <= 0:
        return 0.0
    return float((top_hits / k) / base_rate)


def ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    mu: np.ndarray,
    interval_width: np.ndarray,
    top_fracs: list[float],
    active_threshold: float = 0.5,
) -> dict[str, float]:
    y_true = y_true.astype(int)
    n = len(y_true)
    out: dict[str, float] = {}
    for frac in top_fracs:
        k = max(1, int(n * frac))
        order = np.argsort(-scores)
        top = order[:k]
        pct = int(frac * 100)
        out[f"top_{pct}pct_hit_rate"] = float(y_true[top].mean())
        out[f"ef_{pct}pct"] = enrichment_factor(y_true, scores, frac)

    # High-confidence nomination risk proxy (label-free at selection time)
    confident = (mu > active_threshold) & (interval_width < np.median(interval_width))
    if confident.any():
        hc_err = ((mu > active_threshold) & (y_true == 0))[confident].mean()
        out["high_confidence_error_rate"] = float(hc_err)
    else:
        out["high_confidence_error_rate"] = 0.0

    k = max(1, int(n * 0.01))
    top = np.argsort(-scores)[:k]
    # Observed high-confidence negatives in fixed top-k (evaluation only)
    hc_neg_topk = ((y_true[top] == 0) & (mu[top] > active_threshold)).mean()
    out["hc_negative_rate_at_top1pct"] = float(hc_neg_topk)
    out["hc_nomination_risk_proxy_at_top1pct"] = float(hc_neg_topk)
    return out
