import numpy as np

from oddd.acquisition.batch_selection import _batch_hc_nomination_risk, constrained_batch_select
from oddd.conformal.cluster_conformal import ClusterConformalPredictor
from oddd.conformal.nonconformity import nonconformity_scores
from oddd.data.splits import GroupAssigner, assign_clusters


def test_cluster_conformal_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 200)
    mu = y + rng.normal(0, 0.2, 200)
    scores = nonconformity_scores(y, mu, "regression", normalized=False)
    chains = ["C", "CC", "CCC", "CCCC", "C(C)C"]
    smiles = [f"c1ccccc1C({chains[i % len(chains)]})O" for i in range(200)]
    groups = assign_clusters(smiles, ["scaffold"], min_group_size=5)
    cp = ClusterConformalPredictor(alpha=0.1, min_group_size=5)
    cp.fit(scores, groups)
    interval = cp.predict_interval(mu, groups, "regression")
    covered = (y >= interval.lower) & (y <= interval.upper)
    assert covered.mean() >= 0.85


def test_classification_nonconformity():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.2, 0.8, 0.6, 0.4])
    s = nonconformity_scores(y, p, "classification")
    assert s.shape == (4,)
    assert s[1] < s[0]


def test_group_assigner_maps_calibration_groups_to_test():
    cal_smiles = [f"CC(C)Oc1ccccc1-{i % 4}" for i in range(40)] + [f"CCNCC-{i % 4}" for i in range(40)]
    test_smiles = cal_smiles[:10] + [f"CCCCCC-{i}" for i in range(10)]

    assigner = GroupAssigner(min_group_size=5, seed=0)
    assigner.fit(cal_smiles, ["scaffold"])
    cal_groups, cal_diag = assigner.transform(cal_smiles)
    test_groups, test_diag = assigner.transform(test_smiles)

    assert len(cal_groups) == len(cal_smiles)
    assert len(test_groups) == len(test_smiles)
    assert cal_diag["n_effective_groups"] >= 1
    assert test_diag["mapped_fraction"] > 0.0
    assert (cal_groups != assigner.residual_id).any()
    # Calibration molecules with known scaffold keys should map consistently
    assert cal_groups[0] == test_groups[0]


def test_hc_nomination_risk_modes():
    mu = np.array([0.9, 0.9])
    width = np.array([0.01, 0.99])
    narrow_active = [0]
    wide_active = [1]

    legacy_narrow = _batch_hc_nomination_risk(
        narrow_active, mu, width, mode="legacy_inverted", active_threshold=0.5
    )
    corrected_narrow = _batch_hc_nomination_risk(
        narrow_active, mu, width, mode="high_uncertainty_active", active_threshold=0.5
    )
    corrected_wide = _batch_hc_nomination_risk(
        wide_active, mu, width, mode="high_uncertainty_active", active_threshold=0.5
    )
    none = _batch_hc_nomination_risk(
        narrow_active, mu, width, mode="none", active_threshold=0.5
    )

    assert legacy_narrow == 1.0
    assert corrected_narrow == 0.0
    assert corrected_wide == 1.0
    assert none == 0.0

    n = 20
    rng = np.random.default_rng(0)
    scores = rng.random(n)
    smiles = [f"CCOC{i % 5}" for i in range(n)]
    selected, trace = constrained_batch_select(
        scores,
        smiles,
        rng.random(n),
        rng.random(n),
        rng.random(n),
        rng.random(n) * 0.2,
        budget=5,
        hc_risk_mode="none",
    )
    assert len(selected) == 5
    assert trace["hc_risk_mode"] == "none"
