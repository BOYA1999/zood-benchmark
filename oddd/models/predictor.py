from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = XGBRegressor = None  # type: ignore[misc, assignment]

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None  # type: ignore[misc, assignment]
from torch.utils.data import DataLoader, TensorDataset

from oddd.utils.chemistry import morgan_fp


@dataclass
class PredictorOutput:
    mu: np.ndarray
    sigma: np.ndarray | None = None
    proba: np.ndarray | None = None


class FingerprintMLPPredictor(nn.Module):
    def __init__(self, n_bits: int, hidden_dim: int, dropout: float, task_type: str):
        super().__init__()
        self.task_type = task_type
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _featurize(smiles: list[str], radius: int, n_bits: int) -> np.ndarray:
    return np.stack([morgan_fp(s, radius=radius, n_bits=n_bits) for s in smiles]).astype(np.float32)


def train_predictor(
    smiles: list[str],
    y: np.ndarray,
    task_type: Literal["classification", "regression"],
    cfg: dict,
    seed: int,
) -> tuple[object, np.ndarray]:
    radius = cfg.get("radius", 2)
    n_bits = cfg.get("n_bits", 2048)
    X = _featurize(smiles, radius, n_bits)

    model_type = cfg.get("type", "fingerprint_mlp")

    if model_type == "random_forest":
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        model.fit(X, y)
        return model, X

    if model_type == "xgboost":
        if XGBClassifier is None or XGBRegressor is None:
            raise ImportError("xgboost is not installed")
        common = {
            "n_estimators": cfg.get("n_estimators", 300),
            "max_depth": cfg.get("max_depth", 6),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "subsample": cfg.get("subsample", 0.8),
            "colsample_bytree": cfg.get("colsample_bytree", 0.8),
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if task_type == "classification":
            pos = float(np.sum(y == 1))
            neg = float(np.sum(y == 0))
            scale_pos_weight = neg / max(pos, 1.0)
            model = XGBClassifier(scale_pos_weight=scale_pos_weight, **common)
        else:
            model = XGBRegressor(**common)
        model.fit(X, y)
        return model, X

    if model_type == "lightgbm":
        if LGBMClassifier is None or LGBMRegressor is None:
            raise ImportError("lightgbm is not installed")
        common = {
            "n_estimators": cfg.get("n_estimators", 300),
            "max_depth": cfg.get("max_depth", -1),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "subsample": cfg.get("subsample", 0.8),
            "colsample_bytree": cfg.get("colsample_bytree", 0.8),
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if task_type == "classification":
            pos = float(np.sum(y == 1))
            neg = float(np.sum(y == 0))
            model = LGBMClassifier(scale_pos_weight=neg / max(pos, 1.0), **common)
        else:
            model = LGBMRegressor(**common)
        model.fit(X, y)
        return model, X

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerprintMLPPredictor(
        n_bits=n_bits,
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.2),
        task_type=task_type,
    ).to(device)

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=cfg.get("batch_size", 64), shuffle=True)

    if task_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    model.train()
    for _ in range(cfg.get("epochs", 30)):
        for xb, yb in loader:
            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optim.step()
    model.eval()
    return model, X


def predict(
    model: object,
    X: np.ndarray,
    task_type: str,
) -> PredictorOutput:
    _sklearn_like = (
        RandomForestClassifier,
        RandomForestRegressor,
    )
    if XGBClassifier is not None:
        _sklearn_like = _sklearn_like + (XGBClassifier, XGBRegressor)
    if LGBMClassifier is not None:
        _sklearn_like = _sklearn_like + (LGBMClassifier, LGBMRegressor)

    if isinstance(model, _sklearn_like):
        if task_type == "classification":
            proba = model.predict_proba(X)[:, 1]
            return PredictorOutput(mu=proba, proba=proba, sigma=None)
        mu = model.predict(X)
        return PredictorOutput(mu=mu, sigma=None)

    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    if task_type == "classification":
        proba = 1.0 / (1.0 + np.exp(-logits))
        return PredictorOutput(mu=proba, proba=proba, sigma=None)
    return PredictorOutput(mu=logits, sigma=None)
