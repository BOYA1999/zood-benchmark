import numpy as np

from oddd.acquisition.batch_selection import (
    constrained_batch_select,
    diversity_filtered_top_k_select,
    maxmin_ucb_select,
    random_diverse_topk_select,
    top_k_select,
)
from oddd.metrics.risk import (
    error_detection_auc_from_risk,
    risk_coverage_curve,
    selected_batch_risk_diagnostics,
    similarity_stratified_error_coverage,
    width_decile_error_profile,
)
from oddd.metrics.statistics import (
    bootstrap_ci,
    bootstrap_cohens_d_ci,
    bootstrap_fixed_nomination_ci,
    bootstrap_fixed_topk_ci,
    compare_strategies_bootstrap,
    paired_cohens_d,
    wilcoxon_signed_rank,
)


def test_bootstrap_ci():
    rng = np.random.default_rng(0)
    vals = rng.normal(1.0, 0.2, 100)
    out = bootstrap_ci(vals, n_boot=200, seed=0)
    assert out["ci_low"] < out["point"] < out["ci_high"]


def test_bootstrap_fixed_nomination_ci_contains_point():
    """Fixed nomination bootstrap CI must bracket the plug-in point estimate."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.binomial(1, 0.05, n).astype(int)
    scores = rng.random(n) + y * 0.4
    nominated_idx = np.argsort(-scores)[:20]

    out = bootstrap_fixed_nomination_ci(y, nominated_idx, n_boot=300, seed=0)
    for metric in ("hit_rate", "ef"):
        point = out[metric]["point"]
        assert out[metric]["ci_low"] <= point <= out[metric]["ci_high"]

    topk = bootstrap_fixed_topk_ci(y, scores, top_frac=0.05, n_boot=300, seed=1)
    for metric in ("hit_rate", "ef"):
        point = topk[metric]["point"]
        assert topk[metric]["ci_low"] <= point <= topk[metric]["ci_high"]


def test_batch_selection_fills_budget():
    """Selection strategies should return exactly the requested budget when feasible."""
    rng = np.random.default_rng(0)
    n = 120
    budget = 20
    scores = rng.random(n)
    admet = rng.random(n) * 0.2
    syn = rng.random(n) * 0.2
    mu = rng.random(n)
    width = rng.random(n) * 0.1
    smiles = [f"CCOC{i % 7}" for i in range(n)]

    topk_idx = top_k_select(scores, budget)
    assert len(topk_idx) == budget

    batch_idx, trace = constrained_batch_select(
        scores,
        smiles,
        admet,
        syn,
        mu,
        width,
        budget=budget,
        fill_budget=True,
    )
    assert len(batch_idx) == budget
    assert trace["n_selected"] == budget

    div_idx, div_trace = diversity_filtered_top_k_select(scores, smiles, budget)
    assert len(div_idx) == budget
    assert div_trace["n_selected"] == budget

    mm_idx, mm_trace = maxmin_ucb_select(scores, smiles, budget, alpha_ucb=0.7)
    assert len(mm_idx) == budget
    assert mm_trace["n_selected"] == budget

    rnd_idx, rnd_trace = random_diverse_topk_select(scores, smiles, budget, topk_multiplier=5, seed=0)
    assert len(rnd_idx) == budget
    assert rnd_trace["n_selected"] == budget


def test_cohens_d_and_wilcoxon():
    a = np.array([1.2, 1.3, 1.1, 1.4, 1.25])
    b = np.array([0.8, 0.9, 0.85, 0.95, 0.88])
    d = paired_cohens_d(a, b)
    assert d > 0
    w = wilcoxon_signed_rank(a, b)
    assert w["pvalue"] < 0.1
    d_ci = bootstrap_cohens_d_ci(a, b, n_boot=100, seed=0)
    assert d_ci["ci_low"] > 0


def test_single_point_strategy_comparison_is_not_significance_test():
    out = compare_strategies_bootstrap(
        {
            "odca": {"hit_rate": 0.2},
            "ensemble_ucb": {"hit_rate": 0.4},
        },
        metric_keys=["hit_rate"],
        reference="ensemble_ucb",
        challenger="odca",
    )
    assert out["hit_rate"]["delta"] == -0.2
    assert np.isnan(out["hit_rate"]["fdr_pvalue"])
    assert out["hit_rate"]["test_note"].startswith("not tested")


def test_risk_diagnostics_shapes():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0], dtype=int)
    p = np.array([0.2, 0.8, 0.6, 0.7, 0.55, 0.4, 0.9, 0.1, 0.51, 0.49], dtype=float)
    w = np.linspace(0.05, 0.5, len(y))
    sim = np.linspace(0.2, 0.95, len(y))
    lower = np.clip(p - w / 2, 0.0, 1.0)
    upper = np.clip(p + w / 2, 0.0, 1.0)

    dec = width_decile_error_profile(y, p, w, n_bins=5)
    assert len(dec) == 5
    strat = similarity_stratified_error_coverage(
        y_true=y, pred_prob=p, lower=lower, upper=upper, similarity=sim, n_bins=5
    )
    assert len(strat) == 5
    auc = error_detection_auc_from_risk(y, p, w)
    assert np.isnan(auc) or 0.0 <= auc <= 1.0
    curve = risk_coverage_curve(y, p, w, n_points=5)
    assert len(curve) == 5
    diag = selected_batch_risk_diagnostics(
        y_true=y,
        pred_prob=p,
        width=w,
        similarity=sim,
        selected_idx=np.array([1, 3, 5, 7], dtype=int),
    )
    assert diag["n_selected"] == 4
