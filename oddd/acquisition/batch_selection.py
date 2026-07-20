from __future__ import annotations

from collections import Counter

import numpy as np

from oddd.utils.chemistry import murcko_scaffold


def top_k_select(scores: np.ndarray, budget: int) -> np.ndarray:
    budget = min(int(budget), len(scores))
    order = np.argsort(-scores)
    return order[:budget].astype(np.int64)


def diversity_filtered_top_k_select(
    scores: np.ndarray,
    smiles: list[str],
    budget: int,
    per_round_caps: tuple[int, ...] = (1, 2, 4),
) -> tuple[np.ndarray, dict]:
    """
    Diversity-filtered top-k with phased scaffold-cap relaxation.

    Round 1 allows at most one molecule per scaffold, then relaxes caps in
    successive rounds until the budget is filled; final fallback ignores caps.
    """
    budget = min(int(budget), len(scores))
    order = np.argsort(-scores)
    scaffold_of = [murcko_scaffold(s) for s in smiles]
    selected: list[int] = []
    counts: Counter[str] = Counter()
    final_cap = "unbounded"

    for cap in per_round_caps:
        final_cap = str(cap)
        for idx in order:
            if len(selected) >= budget:
                break
            if int(idx) in selected:
                continue
            scaf = scaffold_of[int(idx)]
            if counts[scaf] >= cap:
                continue
            selected.append(int(idx))
            counts[scaf] += 1
        if len(selected) >= budget:
            break

    if len(selected) < budget:
        for idx in order:
            if len(selected) >= budget:
                break
            if int(idx) in selected:
                continue
            selected.append(int(idx))
        final_cap = "unbounded"

    trace = {
        "selection_mode": "diversity_filtered_top_k",
        "n_selected": int(len(selected)),
        "budget_requested": int(budget),
        "budget_fill_rate": float(len(selected) / budget) if budget else 0.0,
        "final_scaffold_cap": final_cap,
    }
    return np.asarray(selected, dtype=np.int64), trace


def maxmin_ucb_select(
    ucb_scores: np.ndarray,
    smiles: list[str],
    budget: int,
    alpha_ucb: float = 0.7,
) -> tuple[np.ndarray, dict]:
    """
    Greedy scaffold-mismatch UCB: maximize alpha*ucb + (1-alpha)*min_scaffold_mismatch_to_selected.
    Uses a 0/1 scaffold identity distance rather than fingerprint MaxMin.
    """
    budget = min(int(budget), len(ucb_scores))
    n = len(ucb_scores)
    if budget <= 0 or n == 0:
        return np.empty(0, dtype=np.int64), {
            "selection_mode": "maxmin_ucb",
            "n_selected": 0,
            "budget_requested": int(budget),
            "budget_fill_rate": 0.0,
            "alpha_ucb": float(alpha_ucb),
        }

    scaffolds = [murcko_scaffold(s) for s in smiles]
    # Lightweight distance proxy via scaffold mismatch.
    dist = np.ones((n, n), dtype=np.float32)
    for i in range(n):
        dist[i, i] = 0.0
        for j in range(i + 1, n):
            same = float(scaffolds[i] == scaffolds[j])
            d = 1.0 - same
            dist[i, j] = d
            dist[j, i] = d

    ucb = np.asarray(ucb_scores, dtype=float)
    ucb_norm = (ucb - ucb.min()) / (ucb.max() - ucb.min() + 1e-12)
    selected: list[int] = [int(np.argmax(ucb_norm))]
    remaining = set(range(n)) - {selected[0]}

    while len(selected) < budget and remaining:
        rem_list = np.array(sorted(remaining), dtype=int)
        min_dist = dist[np.ix_(rem_list, np.array(selected, dtype=int))].min(axis=1)
        score = alpha_ucb * ucb_norm[rem_list] + (1.0 - alpha_ucb) * min_dist
        best = int(rem_list[int(np.argmax(score))])
        selected.append(best)
        remaining.remove(best)

    trace = {
        "selection_mode": "maxmin_ucb",
        "n_selected": int(len(selected)),
        "budget_requested": int(budget),
        "budget_fill_rate": float(len(selected) / budget) if budget else 0.0,
        "alpha_ucb": float(alpha_ucb),
    }
    return np.asarray(selected, dtype=np.int64), trace


def random_diverse_topk_select(
    scores: np.ndarray,
    smiles: list[str],
    budget: int,
    topk_multiplier: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Random diversity baseline from top-K candidates with scaffold uniqueness pass.
    """
    budget = min(int(budget), len(scores))
    if budget <= 0:
        return np.empty(0, dtype=np.int64), {
            "selection_mode": "random_diverse_topk",
            "n_selected": 0,
            "budget_requested": 0,
            "budget_fill_rate": 0.0,
            "topk_multiplier": int(topk_multiplier),
        }
    order = np.argsort(-scores)
    k = min(len(scores), max(budget, int(topk_multiplier) * budget))
    pool = order[:k].astype(int)
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)

    selected: list[int] = []
    seen_scaffolds: set[str] = set()
    for idx in pool:
        scaf = murcko_scaffold(smiles[int(idx)])
        if scaf in seen_scaffolds:
            continue
        selected.append(int(idx))
        seen_scaffolds.add(scaf)
        if len(selected) >= budget:
            break

    if len(selected) < budget:
        for idx in pool:
            if int(idx) in selected:
                continue
            selected.append(int(idx))
            if len(selected) >= budget:
                break

    trace = {
        "selection_mode": "random_diverse_topk",
        "n_selected": int(len(selected)),
        "budget_requested": int(budget),
        "budget_fill_rate": float(len(selected) / budget) if budget else 0.0,
        "topk_multiplier": int(topk_multiplier),
    }
    return np.asarray(selected, dtype=np.int64), trace


def _batch_hc_nomination_risk(
    trial: list[int],
    mu: np.ndarray,
    interval_width: np.ndarray,
    *,
    mode: str,
    active_threshold: float,
) -> float:
    """
    Label-free batch risk proxy for high-confidence active nominations.

    - ``high_uncertainty_active``: predicted active with *wide* intervals (risk proxy).
    - ``legacy_inverted``: previous implementation (active + *narrow* intervals); audit only.
    - ``none``: disabled (always returns 0).
    """
    if mode == "none" or not trial:
        return 0.0
    active = mu[trial] > active_threshold
    if mode == "legacy_inverted":
        risky = active & (interval_width[trial] < np.median(interval_width))
    elif mode == "high_uncertainty_active":
        risky = active & (interval_width[trial] > np.median(interval_width))
    else:
        raise ValueError(f"Unknown hc_risk_mode: {mode}")
    return float(np.mean(risky))


def constrained_batch_select(
    scores: np.ndarray,
    smiles: list[str],
    admet_risk: np.ndarray,
    syn_risk: np.ndarray,
    mu: np.ndarray,
    interval_width: np.ndarray,
    budget: int,
    max_admet_risk: float = 0.35,
    max_syn_risk: float = 0.40,
    max_scaffold_redundancy: float = 0.25,
    max_hc_nomination_risk: float = 0.20,
    hc_risk_mode: str = "high_uncertainty_active",
    active_threshold: float = 0.5,
    fill_budget: bool = True,
) -> tuple[np.ndarray, dict]:
    """
  Greedy batch selection with portfolio constraints.

    Returns selected indices and a trace dict (n_selected, budget_fill_rate,
    relaxation_stage). When fill_budget=True, constraints are relaxed in phases
    so |selected| approaches budget when feasible.
    """
    budget = min(int(budget), len(scores))
    order = np.argsort(-scores)
    selected: list[int] = []
    scaffolds: list[str] = []

    def _hc_nomination_risk(trial: list[int]) -> float:
        return _batch_hc_nomination_risk(
            trial,
            mu,
            interval_width,
            mode=hc_risk_mode,
            active_threshold=active_threshold,
        )

    def _try_add(idx: int, limits: dict) -> bool:
        trial = selected + [int(idx)]
        if admet_risk[trial].mean() > limits["admet"]:
            return False
        if syn_risk[trial].mean() > limits["syn"]:
            return False
        scaf = murcko_scaffold(smiles[idx])
        trial_scaffolds = scaffolds + [scaf]
        redundancy = 1.0 - len(set(trial_scaffolds)) / len(trial_scaffolds)
        if redundancy > limits["redundancy"]:
            return False
        if _hc_nomination_risk(trial) > limits["hc_risk"]:
            return False
        selected.append(int(idx))
        scaffolds.append(scaf)
        return True

    phases = [
        {"admet": max_admet_risk, "syn": max_syn_risk, "redundancy": max_scaffold_redundancy, "hc_risk": max_hc_nomination_risk, "stage": "strict"},
    ]
    if fill_budget:
        phases.extend([
            {"admet": max_admet_risk, "syn": max_syn_risk, "redundancy": min(0.5, max_scaffold_redundancy + 0.25), "hc_risk": max_hc_nomination_risk, "stage": "relax_redundancy"},
            {"admet": max_admet_risk, "syn": max_syn_risk, "redundancy": 1.0, "hc_risk": max_hc_nomination_risk, "stage": "relax_redundancy_full"},
            {"admet": 1.0, "syn": 1.0, "redundancy": 1.0, "hc_risk": 1.0, "stage": "score_only_fill"},
        ])

    relaxation_stage = "strict"
    for limits in phases:
        relaxation_stage = limits["stage"]
        for idx in order:
            if len(selected) >= budget:
                break
            if idx in selected:
                continue
            _try_add(int(idx), limits)
        if len(selected) >= budget:
            break

    trace = {
        "n_selected": int(len(selected)),
        "budget_requested": int(budget),
        "budget_fill_rate": float(len(selected) / budget) if budget else 0.0,
        "relaxation_stage": relaxation_stage,
        "hc_risk_mode": hc_risk_mode,
    }
    return np.asarray(selected, dtype=np.int64), trace
