from __future__ import annotations

import numpy as np


def width_decile_error_profile(
    y_true: np.ndarray,
    pred_prob: np.ndarray,
    width: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    pred_prob = np.asarray(pred_prob, dtype=float)
    width = np.asarray(width, dtype=float)
    err = (pred_prob >= 0.5).astype(int) != y_true
    qs = np.quantile(width, np.linspace(0, 1, n_bins + 1))
    out: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = float(qs[i]), float(qs[i + 1])
        if i == n_bins - 1:
            mask = (width >= lo) & (width <= hi)
        else:
            mask = (width >= lo) & (width < hi)
        if not np.any(mask):
            out.append({"bin": i + 1, "n": 0, "width_lo": lo, "width_hi": hi, "error_rate": np.nan})
            continue
        out.append(
            {
                "bin": i + 1,
                "n": int(mask.sum()),
                "width_lo": lo,
                "width_hi": hi,
                "error_rate": float(err[mask].mean()),
            }
        )
    return out


def similarity_stratified_error_coverage(
    *,
    y_true: np.ndarray,
    pred_prob: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    similarity: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    pred_prob = np.asarray(pred_prob, dtype=float)
    similarity = np.asarray(similarity, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    pred = (pred_prob >= 0.5).astype(int)
    err = pred != y_true
    covered = (y_true >= lower) & (y_true <= upper)
    qs = np.quantile(similarity, np.linspace(0, 1, n_bins + 1))
    out: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = float(qs[i]), float(qs[i + 1])
        if i == n_bins - 1:
            mask = (similarity >= lo) & (similarity <= hi)
        else:
            mask = (similarity >= lo) & (similarity < hi)
        if not np.any(mask):
            out.append(
                {
                    "bin": i + 1,
                    "n": 0,
                    "similarity_lo": lo,
                    "similarity_hi": hi,
                    "error_rate": np.nan,
                    "coverage": np.nan,
                }
            )
            continue
        out.append(
            {
                "bin": i + 1,
                "n": int(mask.sum()),
                "similarity_lo": lo,
                "similarity_hi": hi,
                "error_rate": float(err[mask].mean()),
                "coverage": float(covered[mask].mean()),
            }
        )
    return out


def error_detection_auc_from_risk(y_true: np.ndarray, pred_prob: np.ndarray, risk: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(pred_prob, dtype=float) >= 0.5).astype(int)
    err = (pred != y_true).astype(int)
    risk = np.asarray(risk, dtype=float)
    if err.min() == err.max():
        return float("nan")
    pos = risk[err == 1]
    neg = risk[err == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U interpretation of AUROC
    wins = 0.0
    for p in pos:
        wins += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
    return float(wins / (len(pos) * len(neg)))


def risk_coverage_curve(
    y_true: np.ndarray,
    pred_prob: np.ndarray,
    risk: np.ndarray,
    n_points: int = 20,
) -> list[dict[str, float]]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(pred_prob, dtype=float) >= 0.5).astype(int)
    err = (pred != y_true).astype(int)
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(risk)
    out: list[dict[str, float]] = []
    for frac in np.linspace(0.1, 1.0, n_points):
        k = max(1, int(len(risk) * frac))
        keep = order[:k]
        out.append(
            {
                "coverage": float(k / len(risk)),
                "risk": float(err[keep].mean()),
            }
        )
    return out


def selected_batch_risk_diagnostics(
    *,
    y_true: np.ndarray,
    pred_prob: np.ndarray,
    width: np.ndarray,
    similarity: np.ndarray,
    selected_idx: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred_prob = np.asarray(pred_prob, dtype=float)
    width = np.asarray(width, dtype=float)
    similarity = np.asarray(similarity, dtype=float)
    idx = np.asarray(selected_idx, dtype=int)
    if len(idx) == 0:
        return {
            "n_selected": 0,
            "false_positive_rate": np.nan,
            "mean_width": np.nan,
            "mean_similarity": np.nan,
            "low_similarity_fraction": np.nan,
        }
    pred = (pred_prob[idx] >= 0.5).astype(int)
    false_positive = (pred == 1) & (y_true[idx] == 0)
    return {
        "n_selected": int(len(idx)),
        "false_positive_rate": float(false_positive.mean()),
        "mean_width": float(width[idx].mean()),
        "mean_similarity": float(similarity[idx].mean()),
        "low_similarity_fraction": float((similarity[idx] < np.median(similarity)).mean()),
    }
