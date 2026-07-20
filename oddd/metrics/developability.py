from __future__ import annotations

import numpy as np

from oddd.utils.chemistry import murcko_scaffold


def developability_metrics(
    selected_idx: np.ndarray,
    smiles: list[str],
    admet_risk: np.ndarray,
    syn_risk: np.ndarray,
    y_true: np.ndarray,
    feasibility_pass: np.ndarray | None = None,
    syn_pass_threshold: float = 0.45,
    admet_pass_threshold: float = 0.35,
) -> dict[str, float]:
    if len(selected_idx) == 0:
        out = {
            "admet_pass_rate": 0.0,
            "syn_pass_rate": 0.0,
            "hit_rate": 0.0,
            "scaffold_diversity": 0.0,
            "novelty_ratio": 0.0,
        }
        if feasibility_pass is not None:
            out["feasibility_pass_rate"] = 0.0
        return out
    scaffolds = [murcko_scaffold(smiles[i]) for i in selected_idx]
    out = {
        "admet_pass_rate": float((admet_risk[selected_idx] <= admet_pass_threshold).mean()),
        "syn_pass_rate": float((syn_risk[selected_idx] <= syn_pass_threshold).mean()),
        "hit_rate": float(y_true[selected_idx].mean()),
        "scaffold_diversity": float(len(set(scaffolds)) / len(scaffolds)),
        "novelty_ratio": float(len(set(scaffolds)) / max(len(set(murcko_scaffold(s) for s in smiles)), 1)),
    }
    if feasibility_pass is not None:
        out["feasibility_pass_rate"] = float(feasibility_pass[selected_idx].mean())
    return out
