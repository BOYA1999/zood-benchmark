from __future__ import annotations

import numpy as np

from oddd.models.predictor import predict, train_predictor


class DeepEnsemble:
    def __init__(self, n_members: int = 5, predictor_cfg: dict | None = None):
        self.n_members = n_members
        self.predictor_cfg = predictor_cfg or {}
        self.members: list = []

    def fit(self, smiles: list[str], y: np.ndarray, task_type: str, seed: int) -> "DeepEnsemble":
        self.members = []
        for m in range(self.n_members):
            model, _ = train_predictor(smiles, y, task_type, self.predictor_cfg, seed=seed + m)
            self.members.append(model)
        return self

    def predict(self, smiles: list[str], task_type: str, predictor_cfg: dict | None = None):
        from oddd.models.predictor import _featurize

        cfg = predictor_cfg or self.predictor_cfg
        radius = cfg.get("radius", 2)
        n_bits = cfg.get("n_bits", 2048)
        X = _featurize(smiles, radius, n_bits)
        preds = []
        for model in self.members:
            out = predict(model, X, task_type)
            preds.append(out.mu)
        arr = np.stack(preds, axis=0)
        mu = arr.mean(axis=0)
        sigma = arr.std(axis=0)
        return mu, sigma
