"""Seeded synthetic observations with additive, constant-variance noise."""

from dataclasses import dataclass

import numpy as np

from .model import langmuir, positive_finite


@dataclass(frozen=True)
class Dataset:
    concentration: np.ndarray
    observed: np.ndarray
    true_quantity: np.ndarray
    K: float
    q_max: float
    sigma: float


def generate_data(
    K: float = 2.0,
    q_max: float = 10.0,
    c_min: float = 0.005,
    c_max: float = 50.0,
    samples: int = 30,
    noise: float = 0.03,
    seed: int | np.random.SeedSequence | None = 42,
) -> Dataset:
    """Sample log-spaced concentrations; noise is sigma / q_max.

    Concentration is treated as exact. Negative noisy observations are retained.
    """
    positive_finite(K, "K")
    positive_finite(q_max, "q_max")
    positive_finite(c_min, "c_min")
    positive_finite(c_max, "c_max")
    if c_max <= c_min:
        raise ValueError("c_max must exceed c_min")
    if not isinstance(samples, (int, np.integer)) or samples < 3:
        raise ValueError("samples must be an integer of at least 3")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    concentration = np.geomspace(c_min, c_max, samples)
    truth = langmuir(concentration, K, q_max)
    sigma = noise * q_max
    if not np.isfinite(sigma):
        raise ValueError("noise * q_max must be finite")
    observed = truth + np.random.default_rng(seed).normal(0, sigma, samples)
    return Dataset(concentration, observed, truth, K, q_max, sigma)
