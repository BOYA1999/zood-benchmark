from __future__ import annotations

import numpy as np


def coverage_aware_score(
    mu: np.ndarray,
    interval_width: np.ndarray,
    novelty: np.ndarray,
    admet_risk: np.ndarray,
    syn_risk: np.ndarray,
    lambda_conf: float = 1.0,
    beta_novelty: float = 0.3,
    gamma_admet: float = 0.5,
    eta_syn: float = 0.4,
) -> np.ndarray:
    """
    A(x) = mu(x) - lambda * w_conf(x) + beta * novelty(x)
           - gamma * r_ADMET(x) - eta * r_syn(x)
    """
    return (
        mu
        - lambda_conf * interval_width
        + beta_novelty * novelty
        - gamma_admet * admet_risk
        - eta_syn * syn_risk
    )


def ensemble_ucb(mu: np.ndarray, sigma: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return mu + beta * sigma


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_so_far: float, xi: float = 0.01) -> np.ndarray:
    from scipy.stats import norm

    sigma = np.maximum(sigma, 1e-9)
    imp = mu - best_so_far - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)
