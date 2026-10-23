"""The independent-site, single-occupancy Langmuir equilibrium model."""

import numpy as np


def positive_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def coverage(concentration, K: float):
    """Return fractional coverage; K has reciprocal concentration units."""
    positive_finite(K, "K")
    concentration = np.asarray(concentration, dtype=float)
    if np.any(~np.isfinite(concentration)) or np.any(concentration < 0):
        raise ValueError("concentration must be finite and nonnegative")
    with np.errstate(over="ignore"):
        x = K * concentration
    # This form also approaches one correctly when K*C overflows.
    return np.divide(x, 1 + x, out=np.ones_like(x), where=np.isfinite(x))


def langmuir(concentration, K: float, q_max: float):
    """Return equilibrium adsorbed quantity in the units of q_max."""
    positive_finite(q_max, "q_max")
    return q_max * coverage(concentration, K)
