from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from oddd.models.predictor import PredictorOutput


@dataclass
class ChempropTrainConfig:
    epochs: int = 15
    batch_size: int = 64
    message_hidden_dim: int = 300
    depth: int = 3
    dropout: float = 0.0
    accelerator: str = "cpu"


def _build_datapoints(smiles: list[str], y: np.ndarray | None = None):
    from chemprop import data

    if y is None:
        return [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]
    return [data.MoleculeDatapoint.from_smi(smi, [float(v)]) for smi, v in zip(smiles, y)]


def train_chemprop(
    train_smiles: list[str],
    y_train: np.ndarray,
    task_type: Literal["classification", "regression"],
    cfg: dict,
    seed: int,
) -> object:
    if task_type != "classification":
        raise ValueError("Chemprop integration currently supports classification only")

    from chemprop import data, featurizers, models, nn
    from lightning import pytorch as pl

    train_cfg = ChempropTrainConfig(
        epochs=int(cfg.get("epochs", 15)),
        batch_size=int(cfg.get("batch_size", 64)),
        message_hidden_dim=int(cfg.get("message_hidden_dim", 300)),
        depth=int(cfg.get("depth", 3)),
        dropout=float(cfg.get("dropout", 0.0)),
        accelerator=str(cfg.get("accelerator", "cpu")),
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_pts = _build_datapoints(train_smiles, y_train)
    train_dset = data.MoleculeDataset(train_pts, featurizer=featurizer)
    train_loader = data.build_dataloader(train_dset, batch_size=train_cfg.batch_size, shuffle=True)

    mp = nn.BondMessagePassing(d_h=train_cfg.message_hidden_dim, depth=train_cfg.depth, dropout=train_cfg.dropout)
    agg = nn.MeanAggregation()
    ffn = nn.BinaryClassificationFFN(n_tasks=1)
    model = models.MPNN(mp, agg, ffn, None, None)

    trainer = pl.Trainer(
        max_epochs=train_cfg.epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator=train_cfg.accelerator,
        devices=1,
        deterministic=True,
    )
    trainer.fit(model, train_loader)
    return {"model": model, "featurizer": featurizer, "trainer": trainer}


def predict_chemprop(bundle: object, smiles: list[str], task_type: str) -> PredictorOutput:
    from chemprop import data

    model = bundle["model"]
    featurizer = bundle["featurizer"]
    trainer = bundle["trainer"]

    test_pts = _build_datapoints(smiles)
    test_dset = data.MoleculeDataset(test_pts, featurizer=featurizer)
    test_loader = data.build_dataloader(test_dset, shuffle=False)
    preds = trainer.predict(model, test_loader)
    proba = np.concatenate(preds, axis=0).reshape(-1).astype(np.float32)
    return PredictorOutput(mu=proba, proba=proba, sigma=None)
