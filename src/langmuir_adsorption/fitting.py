"""Estimate parameters without access to the generating parameters."""

from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from .model import langmuir


@dataclass(frozen=True)
class FitResult:
    method: str
    status: str
    message: str
    K: float | None = None
    q_max: float | None = None
    rmse: float | None = None
    residual_mean: float | None = None
    residual_std: float | None = None
    runtime_seconds: float = 0.0
    K_std: float | None = None
    q_max_std: float | None = None
    parameter_correlation: float | None = None

    @property
    def valid(self) -> bool:
        return self.status in ("ok", "warning")

    def predict(self, concentration):
        if not self.valid:
            raise ValueError(f"Cannot predict from {self.status} fit: {self.message}")
        return langmuir(concentration, self.K, self.q_max)


def _observations(concentration, quantity):
    c, q = np.asarray(concentration, dtype=float), np.asarray(quantity, dtype=float)
    if c.ndim != 1 or q.ndim != 1 or c.shape != q.shape or len(c) < 3:
        raise ValueError("concentration and quantity must be matching 1D arrays with at least 3 observations")
    if np.any(~np.isfinite(c)) or np.any(~np.isfinite(q)) or np.any(c < 0):
        raise ValueError("observations must be finite and concentrations nonnegative")
    if len(np.unique(c)) < 3 or not np.any(c > 0):
        raise ValueError("at least 3 distinct concentrations are required")
    return c, q


def _result(method, c, q, K, q_max, started, **diagnostics):
    if not np.all(np.isfinite([K, q_max])) or K <= 0 or q_max <= 0:
        return FitResult(method, "unphysical", "Estimated K or q_max is nonpositive or nonfinite",
                         float(K) if np.isfinite(K) else None,
                         float(q_max) if np.isfinite(q_max) else None,
                         runtime_seconds=perf_counter() - started)
    residuals = q - langmuir(c, K, q_max)
    return FitResult(
        method=method, K=float(K), q_max=float(q_max),
        rmse=float(np.sqrt(np.mean(residuals**2))),
        residual_mean=float(np.mean(residuals)),
        residual_std=float(np.std(residuals, ddof=1)),
        runtime_seconds=perf_counter() - started, **diagnostics,
    )


def fit_nonlinear(concentration, quantity, sigma: float | None = None) -> FitResult:
    """Fit q directly with positive bounds and data-derived starting values.

    sigma is the known constant measurement standard deviation. Zero denotes a
    noiseless experiment and uses an unweighted fit. Covariance is only a local
    approximation; empirical recovery across experiments is reported separately.
    """
    c, q = _observations(concentration, quantity)
    if sigma is not None and (not np.isfinite(sigma) or sigma < 0):
        raise ValueError("sigma must be finite and nonnegative")
    started = perf_counter()
    # Scaling improves numerical behavior when physical units differ greatly.
    c_scale = float(np.exp(np.mean(np.log(c[c > 0]))))
    q_scale = float(np.max(np.abs(q)))
    if q_scale == 0 or np.max(q) <= 0:
        return FitResult("nonlinear", "failed", "No positive adsorption signal",
                         runtime_seconds=perf_counter() - started)
    known_sigma = sigma is not None and sigma > 0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", OptimizeWarning)
            parameters, covariance = curve_fit(
                langmuir, c / c_scale, q / q_scale,
                p0=(1.0, max(float(np.max(q) / q_scale), 0.01)),
                bounds=(np.finfo(float).eps, np.inf), method="trf",
                sigma=sigma / q_scale if known_sigma else None,
                absolute_sigma=known_sigma, x_scale="jac", max_nfev=3000,
                ftol=1e-10, xtol=1e-10, gtol=1e-10,
            )
        K, q_max = parameters / np.array([c_scale, 1 / q_scale])
        standard = np.sqrt(np.maximum(np.diag(covariance), 0))
        corr = float(covariance[0, 1] / np.prod(standard)) if np.all(standard > 0) else None
        uncertain = (not np.all(np.isfinite(covariance))
                     or np.any(standard / parameters > 1)
                     or (corr is not None and abs(corr) > 0.99))
        flagged = uncertain or bool(caught)
        return _result(
            "nonlinear", c, q, K, q_max, started,
            status="warning" if flagged else "ok",
            message="Weak parameter constraints; inspect repeated-experiment recovery" if flagged else "Converged",
            K_std=float(standard[0] / c_scale) if np.isfinite(standard[0]) else None,
            q_max_std=float(standard[1] * q_scale) if np.isfinite(standard[1]) else None,
            parameter_correlation=corr if corr is not None and np.isfinite(corr) else None,
        )
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return FitResult("nonlinear", "failed", str(error), runtime_seconds=perf_counter() - started)


def fit_linearized(concentration, quantity) -> FitResult:
    """OLS of C/q = a + b*C; q_max=1/b and K=b/a.

    Finite negative observations remain included. No data-dependent filtering or
    clipping is performed; exact zero denominators make this method undefined.
    """
    c, q = _observations(concentration, quantity)
    started = perf_counter()
    if np.any(q == 0):
        return FitResult("linearized", "failed", "C/q is undefined for zero measured quantity",
                         runtime_seconds=perf_counter() - started)
    c_scale = float(np.max(c))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        transformed = c / q
    if np.any(~np.isfinite(transformed)):
        return FitResult("linearized", "failed", "Nonfinite C/q transformation",
                         runtime_seconds=perf_counter() - started)
    try:
        a, scaled_b = np.linalg.lstsq(np.column_stack((np.ones_like(c), c / c_scale)),
                                    transformed, rcond=None)[0]
        b = scaled_b / c_scale
        if a == 0 or b == 0:
            return FitResult("linearized", "failed", "Zero slope or intercept prevents parameter recovery",
                             runtime_seconds=perf_counter() - started)
        return _result("linearized", c, q, b / a, 1 / b, started,
                       status="ok", message="OLS in transformed space; metrics evaluated in q space")
    except np.linalg.LinAlgError as error:
        return FitResult("linearized", "failed", str(error), runtime_seconds=perf_counter() - started)


def fit_both(concentration, quantity, sigma=None) -> list[FitResult]:
    return [fit_nonlinear(concentration, quantity, sigma), fit_linearized(concentration, quantity)]
