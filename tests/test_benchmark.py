import json

import numpy as np

from oddd.benchmark.matrix import BenchmarkMatrix
from oddd.data.datasets import MolecularDataset
from oddd.data.splits import compute_split_diagnostics
from oddd.feasibility.filters import compute_feasibility_profile
from oddd.models.predictor import PredictorOutput
from scripts import run_pipeline


def test_split_diagnostics_strict_fields():
    scaffolds = ["A", "A", "B", "C", "D", "E", "F"]
    y = np.array([0, 1, 0, 1, 1, 0, 1], dtype=int)
    train_idx = np.array([0, 1, 2, 3])
    cal_idx = np.array([4])
    test_idx = np.array([5, 6])
    diag = compute_split_diagnostics(scaffolds, y, train_idx, cal_idx, test_idx)

    assert diag["pool_n_positives"] == 4
    assert diag["train_n_positives"] == 2
    assert diag["test_singleton_scaffold_ratio"] == 1.0
    assert diag["scaffold_disjoint_train_test"] is True
    assert 0.0 < diag["train_positive_coverage"] <= 1.0


def test_feasibility_profile_independent_fields():
    prof = compute_feasibility_profile("CCO")
    assert "sa_score" in prof
    assert "qed" in prof
    assert "pains_hit" in prof
    assert "brenk_hit" in prof
    assert "feasibility_pass" in prof
    assert isinstance(prof["feasibility_pass"], (bool, np.bool_))


def test_chemprop_only_persists_metrics_summary(tmp_path, monkeypatch):
    dataset = MolecularDataset(
        smiles=["CCO", "CCN", "CCC", "c1ccccc1"],
        y=np.array([0, 1, 0, 1], dtype=int),
        task_type="classification",
    )

    monkeypatch.setattr(run_pipeline, "load_dataset", lambda **_: dataset)
    monkeypatch.setattr(
        run_pipeline,
        "make_split",
        lambda *args, **kwargs: (
            np.array([0, 1]),
            np.array([2]),
            np.array([3]),
            {"source": "test"},
        ),
    )
    monkeypatch.setattr(
        run_pipeline,
        "_load_data_provenance",
        lambda cfg: {
            "data_source": "synthetic",
            "endpoint": "NR-AR",
            "n_records": 4,
            "cache_sha256": "testsha",
        },
    )
    monkeypatch.setattr(
        run_pipeline,
        "classification_metrics",
        lambda y_true, y_pred: {"auroc": 0.77, "auprc": 0.31, "n_test": len(y_true), "positive_rate": float(np.mean(y_true))},
    )
    monkeypatch.setattr(run_pipeline, "expected_calibration_error", lambda y_true, y_pred: 0.05)
    monkeypatch.setattr(
        run_pipeline,
        "ranking_metrics",
        lambda *args, **kwargs: {"top_1pct_hit_rate": 0.5, "ef_1pct": 2.0},
    )

    import oddd.models.chemprop_predictor as chemprop_predictor

    monkeypatch.setattr(chemprop_predictor, "train_chemprop", lambda *args, **kwargs: {"dummy": True})
    monkeypatch.setattr(
        chemprop_predictor,
        "predict_chemprop",
        lambda *args, **kwargs: PredictorOutput(mu=np.array([0.8], dtype=np.float32), proba=np.array([0.8])),
    )

    cfg = {
        "experiment": {"seed": 42, "output_dir": str(tmp_path / "runs")},
        "data": {
            "source": "synthetic",
            "task_type": "classification",
            "split_protocol": "random",
            "test_ratio": 0.25,
            "cal_ratio": 0.25,
            "tox21_endpoint": "NR-AR",
        },
        "benchmark": {"chemprop_only": True},
        "evaluation": {"top_k_fractions": [0.5]},
        "chemprop": {},
    }

    matrix = BenchmarkMatrix(tmp_path / "matrix.parquet")
    summary = run_pipeline.run_single_split(cfg, split_protocol="random", matrix=matrix)

    assert summary["benchmark_baselines"][0]["model"] == "chemprop_v2"
    assert summary["benchmark_baselines"][0]["strategy"] == "prediction_only"
    assert summary["benchmark_baselines"][0]["status"] == "ok"
    assert summary["benchmark_baselines"][0]["metrics"]["auroc"] == 0.77
    assert summary["benchmark_baselines"][0]["metrics"]["auprc"] == 0.31

    metrics_path = tmp_path / "runs" / "split_random" / "metrics_chemprop_summary.json"
    assert metrics_path.exists()
    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    chemprop_entry = persisted["benchmark_baselines"][0]
    assert chemprop_entry["model"] == "chemprop_v2"
    assert chemprop_entry["strategy"] == "prediction_only"
    assert chemprop_entry["status"] == "ok"
    assert persisted["benchmark_baselines"][0]["metrics"]["auroc"] == 0.77
    assert persisted["benchmark_baselines"][0]["metrics"]["auprc"] == 0.31
