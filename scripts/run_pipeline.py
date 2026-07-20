#!/usr/bin/env python
"""
OOD-Calibrated Conformal Acquisition — end-to-end pipeline.

Usage:
    python scripts/run_pipeline.py --config configs/default.yml
    python scripts/run_pipeline.py --config configs/tox21_benchmark.yml --split scaffold
    python scripts/run_benchmark.py --config configs/tox21_benchmark.yml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oddd.acquisition.batch_selection import (
    constrained_batch_select,
    diversity_filtered_top_k_select,
    maxmin_ucb_select,
    random_diverse_topk_select,
    top_k_select,
)
from oddd.acquisition.scores import coverage_aware_score
from oddd.baselines.ensemble import DeepEnsemble
from oddd.baselines.ucb import EnsembleUCB
from oddd.baselines.vanilla_conformal import VanillaConformal
from oddd.benchmark.baselines import BaselineRegistry
from oddd.benchmark.matrix import BenchmarkMatrix
from oddd.conformal.cluster_conformal import ClusterConformalPredictor
from oddd.conformal.nonconformity import nonconformity_scores
from oddd.conformal.ood_routing import OODRouter
from oddd.data.datasets import load_dataset
from oddd.data.splits import GroupAssigner, compute_split_diagnostics, make_split
from oddd.feasibility.filters import compute_feasibility_batch
from oddd.metrics.calibration import (
    coverage_metrics,
    expected_calibration_error,
    interval_width_stats,
)
from oddd.metrics.developability import developability_metrics
from oddd.metrics.prediction import classification_metrics
from oddd.metrics.ranking import ranking_metrics
from oddd.metrics.risk import (
    error_detection_auc_from_risk,
    risk_coverage_curve,
    selected_batch_risk_diagnostics,
    similarity_stratified_error_coverage,
    width_decile_error_profile,
)
from oddd.metrics.statistics import bootstrap_fixed_nomination_ci, bootstrap_fixed_topk_ci
from oddd.models.predictor import predict, train_predictor
from oddd.utils.config import load_config
from oddd.utils.manifest import ExperimentManifest
from oddd.utils.seed import set_seed
from oddd.viz.figures import generate_all_figures

STRATEGY_MODEL_MAP = {
    "proposed": ("mlp", "odca"),
    "ensemble_ucb": ("mlp", "ensemble_ucb"),
    "vanilla_conformal": ("mlp", "vanilla_conformal"),
    "potency_only": ("mlp", "potency_only"),
    "diversity_filtered_ucb": ("mlp", "diversity_filtered_ucb"),
    "maxmin_ucb": ("mlp", "maxmin_ucb"),
    "random_diverse_topk": ("mlp", "random_diverse_topk"),
    "ablation_a0_potency": ("mlp", "ablation_a0_potency"),
    "ablation_a1_potency_plus_novelty": ("mlp", "ablation_a1_potency_plus_novelty"),
    "ablation_a2_potency_minus_width": ("mlp", "ablation_a2_potency_minus_width"),
    "ablation_a3_plus_routing": ("mlp", "ablation_a3_plus_routing"),
    "ablation_a4_property_only": ("mlp", "ablation_a4_property_only"),
    "ablation_a5_no_novelty": ("mlp", "ablation_a5_no_novelty"),
    "ablation_a6_full_odca": ("mlp", "ablation_a6_full_odca"),
    "ablation_hc_cap_none": ("mlp", "ablation_hc_cap_none"),
    "ablation_hc_cap_legacy": ("mlp", "ablation_hc_cap_legacy"),
    "ablation_hc_cap_corrected": ("mlp", "ablation_hc_cap_corrected"),
}


def _resolve_manifest_entry(manifest: dict, endpoint: str, use_full: bool) -> dict:
    datasets = manifest.get("datasets", [])
    full_id = f"tox21_{endpoint}_full"
    sub_id = f"tox21_{endpoint}"
    if use_full:
        for d in datasets:
            if d.get("id") == full_id:
                return d
    for d in datasets:
        if d.get("id") in {sub_id, full_id}:
            return d
    return next(iter(datasets), {})


def _load_data_provenance(data_cfg: dict) -> dict:
    from oddd.data.download import CACHE_ROOT

    source = data_cfg.get("source", "synthetic")
    use_full = bool(data_cfg.get("use_full", False))
    if source == "synthetic":
        return {
            "data_source": "synthetic",
            "data_provenance_status": "demonstration_only",
            "note": "In-memory synthetic pool; not a public benchmark release.",
        }
    cache_manifest = CACHE_ROOT / ("tox21" if source == "tox21" else "chembl") / "manifest.json"
    if cache_manifest.exists():
        with open(cache_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        entry = _resolve_manifest_entry(manifest, data_cfg.get("tox21_endpoint", "NR-AR"), use_full)
        return {
            "data_source": source,
            "data_provenance_status": entry.get("source", "unknown"),
            "provenance_tier": entry.get("provenance_tier", "unknown"),
            "download_url": entry.get("download_url"),
            "cache_manifest": str(cache_manifest),
            "cache_sha256": entry.get("sha256"),
            "cache_path": entry.get("path"),
            "n_records": entry.get("n_records"),
            "n_positives": entry.get("n_positives"),
            "positive_rate": entry.get("positive_rate"),
            "endpoint": entry.get("endpoint"),
            "filter_rules": entry.get("filter_rules"),
            "subsample_seed": entry.get("subsample_seed"),
            "subsample_max_rows": entry.get("subsample_max_rows"),
            "use_full": use_full,
            "note": entry.get("note"),
        }
    return {"data_source": source, "data_provenance_status": "missing_cache_manifest", "use_full": use_full}


def _featurize_smiles(smiles, cfg):
    from oddd.models.predictor import _featurize

    return _featurize(smiles, cfg.get("radius", 2), cfg.get("n_bits", 2048))


def _append_matrix_rows(
    matrix: BenchmarkMatrix | None,
    *,
    run_id: str,
    cfg: dict,
    summary: dict,
    boot_metrics: dict,
    benchmark_baselines: list[dict],
    out_dir: Path,
) -> None:
    if matrix is None:
        return
    seed = cfg["experiment"]["seed"]
    prov = summary["data_provenance"]
    split = summary["split_protocol"]
    split_hash = summary["split_hash"]
    diag = summary["split_diagnostics"]

    for entry in benchmark_baselines:
        model = entry["model"]
        strategy = entry["strategy"]
        status = entry["status"]
        metrics = entry.get("metrics", {})
        batch = entry.get("batch", {})
        boot = entry.get("bootstrap", {})
        matrix.append(
            run_id=run_id,
            data_provenance=prov,
            split_protocol=split,
            split_hash=split_hash,
            seed=seed,
            model=model,
            strategy=strategy,
            status=status,
            failure_reason=entry.get("failure_reason"),
            prediction=metrics,
            coverage=metrics if metrics.get("marginal_coverage") is not None else summary.get("proposed_coverage"),
            strategy_metrics=batch or metrics,
            bootstrap=boot,
            split_diagnostics=diag,
            output_dir=out_dir,
        )


def _run_chemprop_only(
    cfg: dict,
    *,
    protocol: str,
    dataset,
    train_idx,
    test_idx,
    split_hash: str,
    split_diagnostics: dict,
    data_provenance: dict,
    out_dir: Path,
    matrix: BenchmarkMatrix | None,
) -> dict:
    """Fast path: train/evaluate Chemprop only and append matrix row."""
    from oddd.models.chemprop_predictor import predict_chemprop, train_chemprop

    seed = cfg["experiment"]["seed"]
    task_type = cfg["data"]["task_type"]
    eval_cfg = cfg.get("evaluation", {})
    top_fracs = eval_cfg.get("top_k_fractions", [0.01, 0.05])

    train_smiles = [dataset.smiles[i] for i in train_idx]
    test_smiles = [dataset.smiles[i] for i in test_idx]
    y_train = dataset.y[train_idx]
    y_test = dataset.y[test_idx]

    benchmark_baselines: list[dict] = []
    try:
        cp_bundle = train_chemprop(train_smiles, y_train, task_type, cfg.get("chemprop", {}), seed)
        cp_out = predict_chemprop(cp_bundle, test_smiles, task_type)
        cp_pred = classification_metrics(y_test, cp_out.mu)
        cp_pred["ece_test"] = expected_calibration_error(y_test, cp_out.mu)
        cp_rank = ranking_metrics(y_test, cp_out.mu, cp_out.mu, np.zeros_like(cp_out.mu), top_fracs)
        metrics = {**cp_pred, **cp_rank}
        benchmark_baselines.append(
            {
                "model": "chemprop_v2",
                "strategy": "prediction_only",
                "status": "ok",
                "metrics": metrics,
                "batch": {},
                "bootstrap": {},
            }
        )
    except Exception as exc:
        benchmark_baselines.append(
            {
                "model": "chemprop_v2",
                "strategy": "prediction_only",
                "status": "failed",
                "failure_reason": str(exc),
                "metrics": {},
                "batch": {},
                "bootstrap": {},
            }
        )

    summary = {
        "split_protocol": protocol,
        "split_hash": split_hash,
        "split_diagnostics": split_diagnostics,
        "data_provenance": data_provenance,
        "task_type": task_type,
        "benchmark_baselines": benchmark_baselines,
        "chemprop_only": True,
    }

    run_id = f"seed{seed}_{protocol}_{split_hash}"
    endpoint = data_provenance.get("endpoint")
    if endpoint:
        run_id = f"{endpoint}_{run_id}"
    _append_matrix_rows(
        matrix,
        run_id=run_id,
        cfg=cfg,
        summary=summary,
        boot_metrics={},
        benchmark_baselines=benchmark_baselines,
        out_dir=out_dir,
    )
    manifest = ExperimentManifest(out_dir, cfg)
    manifest.save_metrics("chemprop_summary", summary)
    manifest.finalize(summary)
    return summary


def run_single_split(
    cfg: dict,
    split_protocol: str | None = None,
    matrix: BenchmarkMatrix | None = None,
) -> dict:
    seed = cfg["experiment"]["seed"]
    set_seed(seed)

    out_dir = Path(cfg["experiment"]["output_dir"])
    if split_protocol:
        out_dir = out_dir / f"split_{split_protocol}"
    manifest = ExperimentManifest(out_dir, cfg)

    data_cfg = cfg["data"]
    use_full = bool(data_cfg.get("use_full", False))
    n_samples = data_cfg.get("n_samples")
    if use_full:
        n_samples = None

    protocol = split_protocol or data_cfg["split_protocol"]
    dataset = load_dataset(
        source=data_cfg["source"],
        task_type=data_cfg["task_type"],
        n_samples=n_samples,
        seed=seed,
        endpoint=data_cfg.get("tox21_endpoint", "NR-AR"),
        chembl_target=data_cfg.get("chembl_target", "CHEMBL203"),
        chembl_activity=data_cfg.get("chembl_activity", "IC50"),
        use_full=use_full,
    )

    train_idx, cal_idx, test_idx, split_meta = make_split(
        dataset,
        protocol=protocol,
        test_ratio=data_cfg["test_ratio"],
        cal_ratio=data_cfg["cal_ratio"],
        seed=seed,
    )
    split_hash = manifest.record_split(protocol, train_idx, cal_idx, test_idx, split_meta)
    split_diagnostics = compute_split_diagnostics(dataset.scaffold, dataset.y, train_idx, cal_idx, test_idx)
    data_provenance = _load_data_provenance(data_cfg)

    benchmark_cfg = cfg.get("benchmark", {})
    if benchmark_cfg.get("chemprop_only"):
        return _run_chemprop_only(
            cfg,
            protocol=protocol,
            dataset=dataset,
            train_idx=train_idx,
            test_idx=test_idx,
            split_hash=split_hash,
            split_diagnostics=split_diagnostics,
            data_provenance=data_provenance,
            out_dir=out_dir,
            matrix=matrix,
        )

    pred_cfg = cfg["predictor"]
    task_type = data_cfg["task_type"]
    train_rf = benchmark_cfg.get("train_rf", True)

    train_smiles = [dataset.smiles[i] for i in train_idx]
    cal_smiles = [dataset.smiles[i] for i in cal_idx]
    test_smiles = [dataset.smiles[i] for i in test_idx]
    y_train = dataset.y[train_idx]
    y_cal = dataset.y[cal_idx]
    y_test = dataset.y[test_idx]

    test_feasibility = compute_feasibility_batch(test_smiles)
    feasibility_pass = test_feasibility["feasibility_pass"].to_numpy(dtype=bool)

    model, X_train = train_predictor(train_smiles, y_train, task_type, pred_cfg, seed)
    X_cal = _featurize_smiles(cal_smiles, pred_cfg)
    X_test = _featurize_smiles(test_smiles, pred_cfg)

    train_out = predict(model, X_train, task_type)
    cal_out = predict(model, X_cal, task_type)
    test_out = predict(model, X_test, task_type)

    prediction_models: dict[str, dict] = {}
    benchmark_baselines: list[dict] = []
    eval_cfg = cfg.get("evaluation", {})
    top_fracs = eval_cfg.get("top_k_fractions", [0.01, 0.05])

    if task_type == "classification":
        mlp_pred = classification_metrics(y_test, test_out.mu)
        mlp_pred["ece_test"] = expected_calibration_error(y_test, test_out.mu)
        prediction_models["mlp"] = mlp_pred
        benchmark_baselines.append(
            {
                "model": "mlp",
                "strategy": "prediction_only",
                "status": "ok",
                "metrics": mlp_pred,
                "batch": {},
                "bootstrap": {},
            }
        )

    if train_rf and task_type == "classification":
        rf_cfg = {**pred_cfg, "type": "random_forest"}
        rf_model, X_train_rf = train_predictor(train_smiles, y_train, task_type, rf_cfg, seed)
        rf_out = predict(rf_model, _featurize_smiles(test_smiles, rf_cfg), task_type)
        rf_pred = classification_metrics(y_test, rf_out.mu)
        rf_pred["ece_test"] = expected_calibration_error(y_test, rf_out.mu)
        rf_rank = ranking_metrics(y_test, rf_out.mu, rf_out.mu, np.zeros_like(rf_out.mu), top_fracs)
        prediction_models["rf"] = {**rf_pred, **rf_rank}
        benchmark_baselines.append(
            {
                "model": "rf",
                "strategy": "prediction_only",
                "status": "ok",
                "metrics": prediction_models["rf"],
                "batch": {},
                "bootstrap": {},
            }
        )

    for booster_name, booster_type in (("xgboost", "xgboost"), ("lightgbm", "lightgbm")):
        if not benchmark_cfg.get(f"train_{booster_name}", True):
            continue
        if task_type != "classification":
            continue
        try:
            booster_cfg = {**pred_cfg, "type": booster_type}
            booster_model, _ = train_predictor(train_smiles, y_train, task_type, booster_cfg, seed)
            booster_out = predict(booster_model, _featurize_smiles(test_smiles, booster_cfg), task_type)
            booster_pred = classification_metrics(y_test, booster_out.mu)
            booster_pred["ece_test"] = expected_calibration_error(y_test, booster_out.mu)
            booster_rank = ranking_metrics(
                y_test, booster_out.mu, booster_out.mu, np.zeros_like(booster_out.mu), top_fracs
            )
            prediction_models[booster_name] = {**booster_pred, **booster_rank}
            benchmark_baselines.append(
                {
                    "model": booster_name,
                    "strategy": "prediction_only",
                    "status": "ok",
                    "metrics": prediction_models[booster_name],
                    "batch": {},
                    "bootstrap": {},
                }
            )
        except ImportError as exc:
            benchmark_baselines.append(
                {
                    "model": booster_name,
                    "strategy": "prediction_only",
                    "status": "blocked",
                    "failure_reason": str(exc),
                    "metrics": {},
                    "batch": {},
                    "bootstrap": {},
                }
            )

    chemprop_endpoints = set(benchmark_cfg.get("chemprop_endpoints", []))
    endpoint_name = data_cfg.get("tox21_endpoint", "NR-AR")
    train_chemprop = bool(benchmark_cfg.get("train_chemprop", False)) and endpoint_name in chemprop_endpoints

    if train_chemprop and task_type == "classification":
        try:
            from oddd.models.chemprop_predictor import predict_chemprop, train_chemprop

            chemprop_cfg = cfg.get("chemprop", {})
            cp_bundle = train_chemprop(train_smiles, y_train, task_type, chemprop_cfg, seed)
            cp_out = predict_chemprop(cp_bundle, test_smiles, task_type)
            cp_pred = classification_metrics(y_test, cp_out.mu)
            cp_pred["ece_test"] = expected_calibration_error(y_test, cp_out.mu)
            cp_rank = ranking_metrics(y_test, cp_out.mu, cp_out.mu, np.zeros_like(cp_out.mu), top_fracs)
            prediction_models["chemprop_v2"] = {**cp_pred, **cp_rank}
            benchmark_baselines.append(
                {
                    "model": "chemprop_v2",
                    "strategy": "prediction_only",
                    "status": "ok",
                    "metrics": prediction_models["chemprop_v2"],
                    "batch": {},
                    "bootstrap": {},
                }
            )
        except Exception as exc:
            benchmark_baselines.append(
                {
                    "model": "chemprop_v2",
                    "strategy": "prediction_only",
                    "status": "failed",
                    "failure_reason": str(exc),
                    "metrics": {},
                    "batch": {},
                    "bootstrap": {},
                }
            )
    else:
        chemprop_probe = BaselineRegistry.probe_chemprop_v2()
        if chemprop_probe.status == "blocked":
            chemprop_probe.failure_reason = (
                chemprop_probe.failure_reason
                or f"chemprop not enabled for endpoint {endpoint_name}"
            )
        benchmark_baselines.append(chemprop_probe.to_log_dict())

    ens_cfg = {**pred_cfg, "epochs": max(10, pred_cfg.get("epochs", 30) // 2)}
    ensemble = DeepEnsemble(n_members=5, predictor_cfg=ens_cfg).fit(train_smiles, y_train, task_type, seed)
    mu_ens_test, sigma_ens_test = ensemble.predict(test_smiles, task_type, predictor_cfg=ens_cfg)

    conf_cfg = cfg["conformal"]
    cluster_sources = conf_cfg.get("cluster_sources", ["scaffold", "fingerprint"])
    group_assigner = GroupAssigner(
        min_group_size=conf_cfg.get("min_group_size", 15),
        seed=seed,
    )
    group_assigner.fit(cal_smiles, cluster_sources)
    cal_groups, cal_group_diag = group_assigner.transform(cal_smiles)
    test_groups, test_group_diag = group_assigner.transform(test_smiles)
    group_diagnostics = {
        "calibration": cal_group_diag,
        "test": test_group_diag,
    }

    cal_scores = nonconformity_scores(
        y_cal,
        cal_out.mu,
        task_type,
        sigma=None,
        normalized=conf_cfg.get("normalized_score", True) if task_type == "regression" else False,
    )

    cp = ClusterConformalPredictor(
        alpha=conf_cfg["alpha"],
        min_group_size=conf_cfg.get("min_group_size", 15),
        conservative_multiplier=conf_cfg.get("ood_routing", {}).get("conservative_multiplier", 1.5),
    )
    cp.fit(cal_scores, cal_groups)

    routed_groups = test_groups
    novelty = np.zeros(len(test_smiles), dtype=np.float32)
    routing_diagnostics: dict = {"routing_enabled": False}
    routing_cfg = conf_cfg.get("ood_routing", {})
    if routing_cfg.get("enabled", True):
        routing_mode = routing_cfg.get("routing_mode", "similarity_only")
        use_scaffold_novelty = routing_cfg.get(
            "use_scaffold_novelty",
            protocol != "scaffold",
        )
        router = OODRouter(
            tanimoto_threshold=routing_cfg.get("tanimoto_threshold", 0.35),
            ood_group_id=cp.ood_group_id_,
            routing_mode=routing_mode,
            use_scaffold_novelty=bool(use_scaffold_novelty),
        )
        router.fit(cal_smiles, cal_groups)
        routing = router.route(test_smiles, test_groups)
        routed_groups = routing.routed_groups
        novelty = routing.novelty
        n_test = max(len(test_smiles), 1)
        routing_diagnostics = {
            "routing_enabled": True,
            "routing_mode": routing_mode,
            "use_scaffold_novelty": bool(use_scaffold_novelty),
            "tanimoto_threshold": float(routing_cfg.get("tanimoto_threshold", 0.35)),
            "ood_routed_fraction": float(routing.is_ood.mean()),
            "similarity_routed_fraction": float(routing.routed_by_similarity.mean()),
            "scaffold_novelty_routed_fraction": float(routing.routed_by_scaffold_novelty.mean()),
            "non_routed_fraction": float((~routing.is_ood).mean()),
        }

    proposed_interval = cp.predict_interval(test_out.mu, routed_groups, task_type)

    vanilla = VanillaConformal(
        alpha=conf_cfg["alpha"],
        normalized=conf_cfg.get("normalized_score", True),
    )
    vanilla.fit(y_cal, cal_out.mu, None, task_type)
    vanilla_interval = vanilla.predict(test_out.mu, task_type)

    acq_cfg = cfg["acquisition"]
    admet_test = dataset.admet_risk[test_idx]
    syn_test = dataset.syn_risk[test_idx]
    lambda_conf = float(acq_cfg["lambda_conf"])
    beta_novelty = float(acq_cfg["beta_novelty"])
    gamma_admet = float(acq_cfg["gamma_admet"])
    eta_syn = float(acq_cfg["eta_syn"])
    vanilla_width = vanilla_interval.width
    routed_width = proposed_interval.width

    def _odca_score(
        *,
        use_width: np.ndarray,
        novelty_weight: float,
        width_weight: float,
        admet_weight: float,
        syn_weight: float,
    ) -> np.ndarray:
        return coverage_aware_score(
            mu=test_out.mu,
            interval_width=use_width,
            novelty=novelty,
            admet_risk=admet_test,
            syn_risk=syn_test,
            lambda_conf=width_weight,
            beta_novelty=novelty_weight,
            gamma_admet=admet_weight,
            eta_syn=syn_weight,
        )

    proposed_scores = _odca_score(
        use_width=routed_width,
        novelty_weight=beta_novelty,
        width_weight=lambda_conf,
        admet_weight=gamma_admet,
        syn_weight=eta_syn,
    )
    ucb_scores = EnsembleUCB(beta=1.0).score(mu_ens_test, sigma_ens_test)
    potency_only = test_out.mu.copy()
    ablation_scores = {
        "ablation_a0_potency": potency_only.copy(),
        "ablation_a1_potency_plus_novelty": _odca_score(
            use_width=np.zeros_like(routed_width),
            novelty_weight=beta_novelty,
            width_weight=0.0,
            admet_weight=0.0,
            syn_weight=0.0,
        ),
        "ablation_a2_potency_minus_width": _odca_score(
            use_width=vanilla_width,
            novelty_weight=0.0,
            width_weight=lambda_conf,
            admet_weight=0.0,
            syn_weight=0.0,
        ),
        "ablation_a3_plus_routing": _odca_score(
            use_width=routed_width,
            novelty_weight=0.0,
            width_weight=lambda_conf,
            admet_weight=0.0,
            syn_weight=0.0,
        ),
        "ablation_a4_property_only": _odca_score(
            use_width=np.zeros_like(routed_width),
            novelty_weight=0.0,
            width_weight=0.0,
            admet_weight=gamma_admet,
            syn_weight=eta_syn,
        ),
        "ablation_a5_no_novelty": _odca_score(
            use_width=routed_width,
            novelty_weight=0.0,
            width_weight=lambda_conf,
            admet_weight=gamma_admet,
            syn_weight=eta_syn,
        ),
        "ablation_a6_full_odca": proposed_scores.copy(),
    }

    constraints = acq_cfg.get("batch_constraints", {})
    budget = acq_cfg["budget"]
    hc_risk_mode = constraints.get("hc_risk_mode", "high_uncertainty_active")
    hc_cap_limit = constraints.get(
        "max_hc_nomination_risk", constraints.get("max_fp_risk", 0.20)
    )

    def _run_constrained(
        scores: np.ndarray,
        *,
        hc_mode: str,
        trace_key: str,
    ) -> tuple[np.ndarray, dict]:
        idx, trace = constrained_batch_select(
            scores,
            test_smiles,
            admet_test,
            syn_test,
            test_out.mu,
            routed_width,
            budget=budget,
            max_admet_risk=constraints.get("max_admet_risk", 0.35),
            max_syn_risk=constraints.get("max_syn_risk", 0.40),
            max_scaffold_redundancy=constraints.get("max_scaffold_redundancy", 0.25),
            max_hc_nomination_risk=hc_cap_limit,
            hc_risk_mode=hc_mode,
            fill_budget=acq_cfg.get("fill_budget", True),
        )
        trace["constraint_variant"] = trace_key
        return idx, trace

    selected, selection_trace = _run_constrained(
        proposed_scores,
        hc_mode=hc_risk_mode,
        trace_key="default",
    )

    strategy_selections: dict[str, np.ndarray] = {"proposed": selected}
    strategy_traces: dict[str, dict] = {"proposed": selection_trace}

    strategy_selections["vanilla_conformal"] = top_k_select(potency_only, budget)
    strategy_selections["ensemble_ucb"] = top_k_select(ucb_scores, budget)
    strategy_selections["potency_only"] = top_k_select(potency_only, budget)

    div_idx, div_trace = diversity_filtered_top_k_select(ucb_scores, test_smiles, budget)
    strategy_selections["diversity_filtered_ucb"] = div_idx
    strategy_traces["diversity_filtered_ucb"] = div_trace

    mm_idx, mm_trace = maxmin_ucb_select(
        ucb_scores, test_smiles, budget, alpha_ucb=float(acq_cfg.get("maxmin_alpha_ucb", 0.7))
    )
    strategy_selections["maxmin_ucb"] = mm_idx
    strategy_traces["maxmin_ucb"] = mm_trace

    rnd_idx, rnd_trace = random_diverse_topk_select(
        ucb_scores,
        test_smiles,
        budget,
        topk_multiplier=int(acq_cfg.get("random_diverse_topk_multiplier", 10)),
        seed=seed,
    )
    strategy_selections["random_diverse_topk"] = rnd_idx
    strategy_traces["random_diverse_topk"] = rnd_trace

    for ab_name in [
        "ablation_a0_potency",
        "ablation_a1_potency_plus_novelty",
        "ablation_a2_potency_minus_width",
        "ablation_a3_plus_routing",
        "ablation_a4_property_only",
        "ablation_a5_no_novelty",
    ]:
        strategy_selections[ab_name] = top_k_select(ablation_scores[ab_name], budget)

    ab6_idx, ab6_trace = _run_constrained(
        ablation_scores["ablation_a6_full_odca"],
        hc_mode=hc_risk_mode,
        trace_key="ablation_a6",
    )
    strategy_selections["ablation_a6_full_odca"] = ab6_idx
    strategy_traces["ablation_a6_full_odca"] = ab6_trace

    if constraints.get("hc_cap_audit", True):
        hc_none_idx, hc_none_trace = _run_constrained(
            proposed_scores,
            hc_mode="none",
            trace_key="hc_cap_none",
        )
        hc_legacy_idx, hc_legacy_trace = _run_constrained(
            proposed_scores,
            hc_mode="legacy_inverted",
            trace_key="hc_cap_legacy",
        )
        hc_corr_idx, hc_corr_trace = _run_constrained(
            proposed_scores,
            hc_mode="high_uncertainty_active",
            trace_key="hc_cap_corrected",
        )
        strategy_selections["ablation_hc_cap_none"] = hc_none_idx
        strategy_selections["ablation_hc_cap_legacy"] = hc_legacy_idx
        strategy_selections["ablation_hc_cap_corrected"] = hc_corr_idx
        strategy_traces["ablation_hc_cap_none"] = hc_none_trace
        strategy_traces["ablation_hc_cap_legacy"] = hc_legacy_trace
        strategy_traces["ablation_hc_cap_corrected"] = hc_corr_trace

    results = {}
    strategy_specs = [
        ("proposed", proposed_scores, routed_width, routed_groups),
        ("vanilla_conformal", potency_only, vanilla_width, test_groups),
        ("ensemble_ucb", ucb_scores, sigma_ens_test, test_groups),
        ("potency_only", potency_only, np.zeros_like(potency_only), test_groups),
        ("diversity_filtered_ucb", ucb_scores, sigma_ens_test, test_groups),
        ("maxmin_ucb", ucb_scores, sigma_ens_test, test_groups),
        ("random_diverse_topk", ucb_scores, sigma_ens_test, test_groups),
        ("ablation_a0_potency", ablation_scores["ablation_a0_potency"], np.zeros_like(potency_only), test_groups),
        (
            "ablation_a1_potency_plus_novelty",
            ablation_scores["ablation_a1_potency_plus_novelty"],
            np.zeros_like(potency_only),
            test_groups,
        ),
        ("ablation_a2_potency_minus_width", ablation_scores["ablation_a2_potency_minus_width"], vanilla_width, test_groups),
        ("ablation_a3_plus_routing", ablation_scores["ablation_a3_plus_routing"], routed_width, routed_groups),
        ("ablation_a4_property_only", ablation_scores["ablation_a4_property_only"], np.zeros_like(potency_only), test_groups),
        ("ablation_a5_no_novelty", ablation_scores["ablation_a5_no_novelty"], routed_width, routed_groups),
        ("ablation_a6_full_odca", ablation_scores["ablation_a6_full_odca"], routed_width, routed_groups),
    ]
    if constraints.get("hc_cap_audit", True):
        strategy_specs.extend(
            [
                ("ablation_hc_cap_none", proposed_scores, routed_width, routed_groups),
                ("ablation_hc_cap_legacy", proposed_scores, routed_width, routed_groups),
                ("ablation_hc_cap_corrected", proposed_scores, routed_width, routed_groups),
            ]
        )
    for name, scores, width, groups in strategy_specs:
        if name == "proposed":
            cov = coverage_metrics(
                y_test, proposed_interval.lower, proposed_interval.upper, routed_groups, conf_cfg["alpha"]
            )
        elif name == "ablation_a3_plus_routing" or name == "ablation_a5_no_novelty" or name == "ablation_a6_full_odca":
            cov = coverage_metrics(
                y_test, proposed_interval.lower, proposed_interval.upper, routed_groups, conf_cfg["alpha"]
            )
        elif name == "vanilla_conformal":
            cov = coverage_metrics(
                y_test, vanilla_interval.lower, vanilla_interval.upper, test_groups, conf_cfg["alpha"]
            )
        else:
            cov = {}

        rank = ranking_metrics(y_test, scores, test_out.mu, width, top_fracs)
        batch_idx = strategy_selections[name]
        dev = developability_metrics(
            batch_idx,
            test_smiles,
            admet_test,
            syn_test,
            y_test,
            feasibility_pass=feasibility_pass,
        )
        if name in strategy_traces:
            dev.update(strategy_traces[name])
        results[name] = {
            **cov,
            **rank,
            **dev,
            "n_selected": int(len(batch_idx)),
            "budget_requested": int(budget),
            "budget_fill_rate": float(len(batch_idx) / budget) if budget else 0.0,
        }

    if task_type == "classification":
        calib = {
            "ece_train": expected_calibration_error(y_train, train_out.mu),
            "ece_test": expected_calibration_error(y_test, test_out.mu),
            **prediction_models.get("mlp", {}),
        }
    else:
        calib = {}

    proposed_cov = coverage_metrics(
        y_test,
        proposed_interval.lower,
        proposed_interval.upper,
        routed_groups,
        conf_cfg["alpha"],
    )
    width_stats = interval_width_stats(proposed_interval.width, routed_groups)

    boot_n = eval_cfg.get("bootstrap_n", 500)
    boot_alpha = 1.0 - eval_cfg.get("confidence_level", 0.95)
    boot_metrics: dict[str, dict] = {}
    for mname, scores in [
        ("proposed", proposed_scores),
        ("ensemble_ucb", ucb_scores),
        ("diversity_filtered_ucb", ucb_scores),
    ]:
        boot_metrics[mname] = {
            "top_1pct": bootstrap_fixed_topk_ci(
                y_test, scores, top_frac=0.01, n_boot=boot_n, alpha=boot_alpha, seed=seed
            ),
            "batch_b40": bootstrap_fixed_nomination_ci(
                y_test,
                strategy_selections[mname],
                n_boot=boot_n,
                alpha=boot_alpha,
                seed=seed + 1,
            ),
        }

    similarity_to_cal = 1.0 - novelty
    risk_diagnostics = {
        "width_decile_error_profile": width_decile_error_profile(y_test, test_out.mu, routed_width),
        "similarity_stratified_error_coverage": similarity_stratified_error_coverage(
            y_true=y_test,
            pred_prob=test_out.mu,
            lower=proposed_interval.lower,
            upper=proposed_interval.upper,
            similarity=similarity_to_cal,
            n_bins=10,
        ),
        "uncertainty_error_auc": error_detection_auc_from_risk(y_test, test_out.mu, routed_width),
        "risk_coverage_curve": risk_coverage_curve(y_test, test_out.mu, routed_width, n_points=20),
        "selected_batch_profiles": {
            key: selected_batch_risk_diagnostics(
                y_true=y_test,
                pred_prob=test_out.mu,
                width=routed_width,
                similarity=similarity_to_cal,
                selected_idx=strategy_selections[key],
            )
            for key in ["proposed", "ensemble_ucb", "diversity_filtered_ucb", "ablation_a6_full_odca"]
        },
    }

    for strat_key, (model_name, strategy_name) in STRATEGY_MODEL_MAP.items():
        if strat_key not in results:
            continue
        benchmark_baselines.append(
            {
                "model": model_name,
                "strategy": strategy_name,
                "status": "ok",
                "metrics": prediction_models.get(model_name, calib),
                "batch": results[strat_key],
                "bootstrap": boot_metrics.get(strat_key, {}),
            }
        )

    stat_comparison = {}
    if boot_metrics.get("proposed") and boot_metrics.get("ensemble_ucb"):
        prop_ef = boot_metrics["proposed"]["top_1pct"]["ef"]["point"]
        ucb_ef = boot_metrics["ensemble_ucb"]["top_1pct"]["ef"]["point"]
        stat_comparison["ef_1pct"] = {
            "proposed_point": prop_ef,
            "ucb_point": ucb_ef,
            "delta": float(prop_ef - ucb_ef),
            "proposed_ci": boot_metrics["proposed"]["top_1pct"]["ef"],
            "ucb_ci": boot_metrics["ensemble_ucb"]["top_1pct"]["ef"],
            "estimand": "fixed_top1pct_paired_bootstrap",
        }

    summary = {
        "split_protocol": protocol,
        "split_hash": split_hash,
        "split_diagnostics": split_diagnostics,
        "data_provenance": data_provenance,
        "task_type": task_type,
        "calibration": calib,
        "prediction": calib if task_type == "classification" else {},
        "prediction_models": prediction_models,
        "benchmark_baselines": benchmark_baselines,
        "proposed_coverage": proposed_cov,
        "proposed_width": width_stats,
        "strategies": results,
        "bootstrap": boot_metrics,
        "risk_diagnostics": risk_diagnostics,
        "statistical_comparison": stat_comparison,
        "selection_trace": strategy_traces,
        "budget_requested": int(budget),
        "group_diagnostics": group_diagnostics,
        "routing_diagnostics": routing_diagnostics,
        "ood_routed_fraction": float(routing_diagnostics.get("ood_routed_fraction", 0.0)),
        "hc_risk_mode": hc_risk_mode,
    }

    pred_df = pd.DataFrame(
        {
            "smiles": test_smiles,
            "y_true": y_test,
            "mu": test_out.mu,
            "lower": proposed_interval.lower,
            "upper": proposed_interval.upper,
            "width": proposed_interval.width,
            "group_id": routed_groups,
            "novelty": novelty,
            "similarity_to_cal": 1.0 - novelty,
            "admet_risk": dataset.admet_risk[test_idx],
            "syn_risk": dataset.syn_risk[test_idx],
            "proposed_score": proposed_scores,
            "ucb_score": ucb_scores,
            "selected_proposed": np.isin(np.arange(len(test_smiles)), selected),
        }
    )
    for strat_name, idx in strategy_selections.items():
        pred_df[f"selected_{strat_name}"] = np.isin(np.arange(len(test_smiles)), idx)
    for col in test_feasibility.columns:
        pred_df[f"feas_{col}"] = test_feasibility[col].values

    manifest.save_predictions("test", pred_df, {"split_hash": split_hash})
    manifest.save_metrics("summary", summary)
    manifest.finalize(summary)

    run_id = f"seed{seed}_{protocol}_{split_hash}"
    _append_matrix_rows(
        matrix,
        run_id=run_id,
        cfg=cfg,
        summary=summary,
        boot_metrics=boot_metrics,
        benchmark_baselines=benchmark_baselines,
        out_dir=out_dir,
    )

    if cfg.get("figures", {}).get("enabled", True):
        try:
            all_splits = out_dir.parent / "all_splits_summary.json"
            generate_all_figures(out_dir, all_splits if all_splits.exists() else None)
        except Exception as exc:
            print(f"[warn] Figure generation skipped: {exc}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run ODDD conformal acquisition pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yml")
    parser.add_argument("--split", type=str, default=None, help="Override split protocol")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    summary = run_single_split(cfg, split_protocol=args.split)
    print("=== ODDD Pipeline Complete ===")
    print(f"Split: {summary['split_protocol']}  hash={summary['split_hash']}")
    prop = summary["strategies"]["proposed"]
    ucb = summary["strategies"]["ensemble_ucb"]
    print(f"Proposed EF1%: {prop.get('ef_1pct', 0):.3f}  UCB EF1%: {ucb.get('ef_1pct', 0):.3f}")
    print(f"Proposed high-conf error: {prop.get('high_confidence_error_rate', 0):.3f}")
    print(f"Worst-group coverage: {summary['proposed_coverage'].get('worst_group_coverage', 0):.3f}")
    print(f"Results saved to: {cfg['experiment']['output_dir']}")


if __name__ == "__main__":
    main()
