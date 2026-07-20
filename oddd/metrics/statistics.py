from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable = np.mean,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "std": np.nan}

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = len(values)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[b] = statistic(sample)

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {
        "point": float(statistic(values)),
        "ci_low": lo,
        "ci_high": hi,
        "std": float(np.std(boots)),
    }


def paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    sd = np.std(diff, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def bootstrap_cohens_d_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    ds = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ds[i] = paired_cohens_d(a[idx], b[idx])
    point = paired_cohens_d(a, b)
    return {
        "point": point,
        "ci_low": float(np.quantile(ds, alpha / 2)),
        "ci_high": float(np.quantile(ds, 1 - alpha / 2)),
        "interpretation": _interpret_d(point),
    }


def wilcoxon_signed_rank(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if np.allclose(diff, 0):
        return {"statistic": 0.0, "pvalue": 1.0}
    res = stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox")
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, pvalues[order[i]] * n / rank)
        ranked[order[i]] = val
        prev = val
    return np.clip(ranked, 0.0, 1.0)


def compare_strategies_bootstrap(
    strategy_metrics: dict[str, dict[str, float]],
    metric_keys: list[str],
    reference: str,
    challenger: str,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict]:
    """
    Bootstrap comparison between two strategies on repeated-run metric vectors.
    strategy_metrics: {strategy: {metric: value}} for single run,
    or pass arrays via strategy_metric_runs in pipeline.
    """
    out = {}
    for key in metric_keys:
        if key not in strategy_metrics.get(challenger, {}):
            continue
        if key not in strategy_metrics.get(reference, {}):
            continue
        a_val = strategy_metrics[challenger][key]
        b_val = strategy_metrics[reference][key]
        # Single-point fallback: report difference only
        out[key] = {
            "challenger": float(a_val),
            "reference": float(b_val),
            "delta": float(a_val - b_val),
            "fdr_pvalue": np.nan,
            "test_note": "not tested; only single-point strategy summaries were provided",
        }
    return out


def bootstrap_fixed_nomination_ci(
    y_true: np.ndarray,
    nominated_idx: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """
    Paired bootstrap with a fixed nomination set on the test pool.

    The ranking / nominated indices are computed once on the full test set.
    Each bootstrap replicate resamples the nominated labels and the test-pool
    base rate independently with replacement, then recomputes hit rate and EF.
    """
    y_true = np.asarray(y_true).astype(int)
    nominated_idx = np.asarray(nominated_idx, dtype=int)
    n = len(y_true)
    k = len(nominated_idx)
    rng = np.random.default_rng(seed)

    y_nom = y_true[nominated_idx]
    base_rate = float(y_true.mean())
    hit_point = float(y_nom.mean()) if k else 0.0
    ef_point = float((hit_point / base_rate) if base_rate > 0 else 0.0)

    boot_hits, boot_efs = [], []
    for _ in range(n_boot):
        nom_boot = rng.integers(0, k, size=k) if k else np.empty(0, dtype=int)
        pool_boot = rng.integers(0, n, size=n)
        hit_b = float(y_nom[nom_boot].mean()) if k else 0.0
        base_b = float(y_true[pool_boot].mean())
        ef_b = float((hit_b / base_b) if base_b > 0 else 0.0)
        boot_hits.append(hit_b)
        boot_efs.append(ef_b)

    def _ci(vals: list[float], point: float) -> dict[str, float]:
        arr = np.asarray(vals, dtype=float)
        return {
            "point": point,
            "ci_low": float(np.quantile(arr, alpha / 2)),
            "ci_high": float(np.quantile(arr, 1 - alpha / 2)),
            "std": float(np.std(arr)),
        }

    return {
        "hit_rate": _ci(boot_hits, hit_point),
        "ef": _ci(boot_efs, ef_point),
        "estimand": "fixed_nomination_set_paired_bootstrap",
        "n_nominated": int(k),
        "top_frac_equivalent": float(k / n) if n else 0.0,
    }


def bootstrap_fixed_topk_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    top_frac: float,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Fixed top-k (by score) paired bootstrap CI for hit rate and EF."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    k = max(1, int(len(y_true) * top_frac))
    order = np.argsort(-scores)
    nominated_idx = order[:k]
    out = bootstrap_fixed_nomination_ci(y_true, nominated_idx, n_boot=n_boot, alpha=alpha, seed=seed)
    out["top_frac"] = float(top_frac)
    return out


def bootstrap_metric_from_runs(
    runs: list[dict[str, float]],
    metric: str,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    values = np.array([r[metric] for r in runs if metric in r], dtype=float)
    return bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)


def compare_repeated_runs(
    runs_a: list[dict[str, float]],
    runs_b: list[dict[str, float]],
    metric: str,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float | str]:
    a = np.array([r[metric] for r in runs_a if metric in r], dtype=float)
    b = np.array([r[metric] for r in runs_b if metric in r], dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return {"metric": metric, "warning": "insufficient repeated runs"}
    a, b = a[:n], b[:n]
    wil = wilcoxon_signed_rank(a, b)
    d = bootstrap_cohens_d_ci(a, b, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "metric": metric,
        "mean_delta": float(np.mean(a - b)),
        "wilcoxon_p": wil["pvalue"],
        "cohens_d": d["point"],
        "cohens_d_ci_low": d["ci_low"],
        "cohens_d_ci_high": d["ci_high"],
        "interpretation": d["interpretation"],
    }


def _interpret_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"
