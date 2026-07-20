#!/usr/bin/env python
"""Sweep acquisition weights and plot Pareto frontier (enrichment vs developability)."""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oddd.acquisition.batch_selection import constrained_batch_select, top_k_select
from oddd.acquisition.scores import coverage_aware_score
from oddd.metrics.developability import developability_metrics
from oddd.metrics.ranking import ranking_metrics
from oddd.utils.config import load_config
from scripts.run_pipeline import run_single_split

LAMBDA_CONF = [0.0, 0.25, 0.5, 1.0]
GAMMA_ADMET = [0.0, 0.2, 0.4]
ETA_SYN = [0.0, 0.15, 0.3]
METRIC_KEYS = ("ef_1pct", "scaffold_diversity", "hit_rate", "syn_pass_rate")


def _pareto_mask(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Non-dominated mask when both objectives are maximized."""
    n = len(xs)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (xs >= xs[i]) & (ys >= ys[i]) & ((xs > xs[i]) | (ys > ys[i]))
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


def _weight_key(point: dict) -> tuple[float, float, float]:
    return (point["lambda_conf"], point["gamma_admet"], point["eta_syn"])


def _load_or_run(seed: int, split: str, config_path: Path, pack_root: Path) -> tuple[pd.DataFrame, dict]:
    run_dir = pack_root / f"seed_{seed}" / f"split_{split}"
    pred_path = run_dir / "predictions_test.parquet"
    cfg_path = run_dir / "config.json"

    if pred_path.exists() and cfg_path.exists():
        print(f"  reuse predictions: {run_dir}")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return pd.read_parquet(pred_path), cfg

    print(f"  run pipeline: seed={seed} split={split}")
    cfg = load_config(config_path)
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["output_dir"] = str(pack_root / f"seed_{seed}")
    cfg.setdefault("figures", {})["enabled"] = False
    run_single_split(cfg, split_protocol=split)
    with open(run_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    return pd.read_parquet(pred_path), cfg


def _evaluate_point(
    df: pd.DataFrame,
    cfg: dict,
    lambda_conf: float,
    gamma_admet: float,
    eta_syn: float,
) -> dict:
    acq = cfg["acquisition"]
    constraints = acq.get("batch_constraints", {})
    budget = int(acq["budget"])
    beta_novelty = float(acq.get("beta_novelty", 0.2))

    mu = df["mu"].to_numpy(dtype=float)
    width = df["width"].to_numpy(dtype=float)
    novelty = df["novelty"].to_numpy(dtype=float)
    admet = df["admet_risk"].to_numpy(dtype=float)
    syn = df["syn_risk"].to_numpy(dtype=float)
    y = df["y_true"].to_numpy(dtype=int)
    smiles = df["smiles"].astype(str).tolist()

    scores = coverage_aware_score(
        mu, width, novelty, admet, syn,
        lambda_conf=lambda_conf,
        beta_novelty=beta_novelty,
        gamma_admet=gamma_admet,
        eta_syn=eta_syn,
    )
    selected, trace = constrained_batch_select(
        scores,
        smiles,
        admet,
        syn,
        mu,
        width,
        budget=budget,
        max_admet_risk=constraints.get("max_admet_risk", 0.35),
        max_syn_risk=constraints.get("max_syn_risk", 0.40),
        max_scaffold_redundancy=constraints.get("max_scaffold_redundancy", 0.25),
        max_hc_nomination_risk=constraints.get(
            "max_hc_nomination_risk", constraints.get("max_fp_risk", 0.20)
        ),
        hc_risk_mode=constraints.get("hc_risk_mode", "high_uncertainty_active"),
        fill_budget=acq.get("fill_budget", True),
    )
    dev = developability_metrics(selected, smiles, admet, syn, y)
    rank = ranking_metrics(y, scores, mu, width, cfg.get("evaluation", {}).get("top_k_fractions", [0.01, 0.05]))
    return {
        "lambda_conf": lambda_conf,
        "gamma_admet": gamma_admet,
        "eta_syn": eta_syn,
        **trace,
        **dev,
        **rank,
    }


def _baseline_ucb(df: pd.DataFrame, cfg: dict) -> dict:
    budget = int(cfg["acquisition"]["budget"])
    y = df["y_true"].to_numpy(dtype=int)
    mu = df["mu"].to_numpy(dtype=float)
    width = df["width"].to_numpy(dtype=float)
    smiles = df["smiles"].astype(str).tolist()
    admet = df["admet_risk"].to_numpy(dtype=float)
    syn = df["syn_risk"].to_numpy(dtype=float)
    ucb = df["ucb_score"].to_numpy(dtype=float)
    selected = top_k_select(ucb, budget)
    dev = developability_metrics(selected, smiles, admet, syn, y)
    rank = ranking_metrics(y, ucb, mu, width, cfg.get("evaluation", {}).get("top_k_fractions", [0.01, 0.05]))
    return {"method": "ensemble_ucb", **dev, **rank, "n_selected": int(len(selected))}


def _aggregate_by_weight(seed_points: dict[int, list[dict]]) -> list[dict]:
    """Mean/std across seeds for each weight configuration."""
    buckets: dict[tuple[float, float, float], list[dict]] = {}
    for points in seed_points.values():
        for pt in points:
            buckets.setdefault(_weight_key(pt), []).append(pt)

    aggregated = []
    for (lc, ga, es), pts in sorted(buckets.items()):
        row = {"lambda_conf": lc, "gamma_admet": ga, "eta_syn": es, "n_seeds": len(pts)}
        for key in METRIC_KEYS:
            vals = [p[key] for p in pts]
            row[key] = float(stats.mean(vals))
            row[f"{key}_std"] = float(stats.pstdev(vals)) if len(vals) > 1 else 0.0
        aggregated.append(row)
    return aggregated


def _aggregate_baseline(seed_baselines: dict[int, dict]) -> dict:
    out = {"method": "ensemble_ucb", "n_seeds": len(seed_baselines)}
    for key in METRIC_KEYS:
        vals = [b[key] for b in seed_baselines.values()]
        out[key] = float(stats.mean(vals))
        out[f"{key}_std"] = float(stats.pstdev(vals)) if len(vals) > 1 else 0.0
    return out


def _plot_frontier(
    points: list[dict],
    baseline: dict,
    out_dir: Path,
    split: str,
    label: str,
    *,
    xerr: np.ndarray | None = None,
    yerr: np.ndarray | None = None,
    all_seed_points: list[dict] | None = None,
    filename_stem: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ef = np.array([p["ef_1pct"] for p in points], dtype=float)
    div = np.array([p["scaffold_diversity"] for p in points], dtype=float)
    hit = np.array([p["hit_rate"] for p in points], dtype=float)
    syn = np.array([p["syn_pass_rate"] for p in points], dtype=float)
    mask = _pareto_mask(ef, div)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    if all_seed_points:
        axes[0].scatter(
            [p["ef_1pct"] for p in all_seed_points],
            [p["scaffold_diversity"] for p in all_seed_points],
            c="#d0d7de", s=14, alpha=0.45, label="Per-seed grid",
        )
        axes[1].scatter(
            [p["hit_rate"] for p in all_seed_points],
            [p["syn_pass_rate"] for p in all_seed_points],
            c="#d0d7de", s=14, alpha=0.45, label="Per-seed grid",
        )

    if xerr is not None and yerr is not None:
        axes[0].errorbar(
            ef[~mask], div[~mask], xerr=xerr[~mask], yerr=yerr[~mask],
            fmt="o", c="#9aa5b1", ms=5, alpha=0.8, capsize=2, label="ODCA mean",
        )
        axes[0].errorbar(
            ef[mask], div[mask], xerr=xerr[mask], yerr=yerr[mask],
            fmt="o", c="#2c5aa0", ms=7, capsize=2, mew=0.4, mec="k", label="Pareto mean",
        )
    else:
        axes[0].scatter(ef[~mask], div[~mask], c="#9aa5b1", s=28, alpha=0.75, label="ODCA grid")
        axes[0].scatter(ef[mask], div[mask], c="#2c5aa0", s=55, edgecolors="k", linewidths=0.4, label="Pareto")

    axes[0].scatter(
        [baseline["ef_1pct"]], [baseline["scaffold_diversity"]],
        c="#c44e52", marker="*", s=180, label="Ensemble-UCB",
        zorder=5,
    )
    axes[0].set_xlabel("EF@1%")
    axes[0].set_ylabel("Scaffold diversity")
    axes[0].set_title(label)
    axes[0].legend(fontsize=8)

    axes[1].scatter(hit[~mask], syn[~mask], c="#9aa5b1", s=28, alpha=0.75, label="ODCA grid")
    axes[1].scatter(hit[mask], syn[mask], c="#2c5aa0", s=55, edgecolors="k", linewidths=0.4, label="Pareto")
    axes[1].scatter(
        [baseline["hit_rate"]], [baseline["syn_pass_rate"]],
        c="#c44e52", marker="*", s=180, label="Ensemble-UCB",
        zorder=5,
    )
    axes[1].set_xlabel("Batch hit rate (B=40)")
    axes[1].set_ylabel("Syn pass rate")
    axes[1].legend(fontsize=8)

    fig.suptitle("ODCA weight sweep: enrichment vs developability trade-off", fontsize=11, fontweight="bold")
    fig.tight_layout()
    stem = filename_stem or f"pareto_frontier_{split}"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _sweep_split(seed: int, split: str, grid: list[tuple], config_path: Path, pack_root: Path) -> dict:
    df, cfg = _load_or_run(seed, split, config_path, pack_root)
    baseline = _baseline_ucb(df, cfg)
    points = [_evaluate_point(df, cfg, lc, ga, es) for lc, ga, es in grid]
    ef = np.array([p["ef_1pct"] for p in points])
    div = np.array([p["scaffold_diversity"] for p in points])
    pareto_idx = np.where(_pareto_mask(ef, div))[0].tolist()
    return {
        "baseline_ucb": baseline,
        "all_points": points,
        "pareto_indices": pareto_idx,
        "pareto_points": [points[i] for i in pareto_idx],
    }


def main():
    parser = argparse.ArgumentParser(description="Pareto frontier weight sweep")
    parser.add_argument("--config", default="configs/tox21_paper.yml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--splits", nargs="+", default=["scaffold", "cluster"])
    parser.add_argument("--aggregate-only", action="store_true", help="Skip per-seed plots")
    args = parser.parse_args()

    base_cfg = load_config(ROOT / args.config)
    pack_root = Path(base_cfg["experiment"]["output_dir"]) / "rigor_pack"
    out_root = Path(base_cfg["experiment"]["output_dir"]) / "pareto_sweep"
    grid = list(product(LAMBDA_CONF, GAMMA_ADMET, ETA_SYN))
    config_path = ROOT / args.config

    per_seed_results: dict[str, dict] = {}
    aggregated_results: dict[str, dict] = {}

    for seed in args.seeds:
        seed_results: dict[str, dict] = {}
        for split in args.splits:
            print(f"\n>>> seed={seed} split={split}")
            result = _sweep_split(seed, split, grid, config_path, pack_root)
            seed_results[split] = result
            if not args.aggregate_only and len(args.seeds) == 1:
                split_out = out_root / f"seed_{seed}" / f"split_{split}"
                _plot_frontier(
                    result["all_points"],
                    result["baseline_ucb"],
                    split_out,
                    split,
                    f"{split} / seed {seed}",
                )
        per_seed_results[f"seed_{seed}"] = seed_results

    for split in args.splits:
        seed_points = {seed: per_seed_results[f"seed_{seed}"][split]["all_points"] for seed in args.seeds}
        seed_baselines = {seed: per_seed_results[f"seed_{seed}"][split]["baseline_ucb"] for seed in args.seeds}
        all_flat = [pt for pts in seed_points.values() for pt in pts]

        mean_points = _aggregate_by_weight(seed_points)
        mean_baseline = _aggregate_baseline(seed_baselines)
        ef = np.array([p["ef_1pct"] for p in mean_points])
        div = np.array([p["scaffold_diversity"] for p in mean_points])
        ef_std = np.array([p["ef_1pct_std"] for p in mean_points])
        div_std = np.array([p["scaffold_diversity_std"] for p in mean_points])
        pareto_idx = np.where(_pareto_mask(ef, div))[0].tolist()

        agg_out = out_root / "aggregated" / f"split_{split}"
        label = f"{split} / seeds {min(args.seeds)}–{max(args.seeds)} (mean±std, n={len(args.seeds)})"
        _plot_frontier(
            mean_points,
            mean_baseline,
            agg_out,
            split,
            label,
            xerr=ef_std,
            yerr=div_std,
            all_seed_points=all_flat,
            filename_stem=f"pareto_frontier_{split}_aggregated",
        )

        aggregated_results[split] = {
            "seeds": args.seeds,
            "baseline_ucb_mean": mean_baseline,
            "mean_grid": mean_points,
            "pareto_indices": pareto_idx,
            "pareto_points_mean": [mean_points[i] for i in pareto_idx],
            "n_grid_points_total": len(all_flat),
        }
        print(f"\n[aggregated/{split}] pareto points on mean grid: {len(pareto_idx)}")
        print(
            f"  UCB mean: EF1%={mean_baseline['ef_1pct']:.2f} "
            f"div={mean_baseline['scaffold_diversity']:.2f} "
            f"hit={mean_baseline['hit_rate']:.2f} syn={mean_baseline['syn_pass_rate']:.2f}"
        )

    summary_path = out_root / "pareto_sweep_aggregated.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": args.seeds,
                "splits": args.splits,
                "weight_grid": {
                    "lambda_conf": LAMBDA_CONF,
                    "gamma_admet": GAMMA_ADMET,
                    "eta_syn": ETA_SYN,
                },
                "per_seed": per_seed_results,
                "aggregated": aggregated_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nPareto sweep complete: {summary_path}")


if __name__ == "__main__":
    main()
