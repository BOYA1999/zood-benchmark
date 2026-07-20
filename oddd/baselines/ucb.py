from __future__ import annotations

import numpy as np

from oddd.acquisition.scores import ensemble_ucb


class EnsembleUCB:
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def score(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return ensemble_ucb(mu, sigma, beta=self.beta)
