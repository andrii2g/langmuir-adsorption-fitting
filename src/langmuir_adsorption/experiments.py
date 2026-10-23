"""Paired Monte Carlo experiments over concentration coverage and noise."""

from dataclasses import asdict

import numpy as np

from .fitting import fit_both
from .synthetic import generate_data


REGIMES = {
    "low": (0.001, 0.1),
    "transition": (0.1, 10.0),
    "broad": (0.01, 100.0),
    "saturation": (10.0, 100.0),
}


def fit_record(fit, K, q_max):
    record = asdict(fit)
    record["K_error_percent"] = 100 * (fit.K / K - 1) if fit.valid else None
    record["q_max_error_percent"] = 100 * (fit.q_max / q_max - 1) if fit.valid else None
    return record


def run_sweep(K=2.0, q_max=10.0, samples=30, noise_levels=(0.0, 0.01, 0.05, 0.1),
              repeats=100, seed=42):
    """Return one record per fit, including failed and unphysical fits.

    A separate random stream generates each dataset; both methods see that exact
    dataset. True K sets the experimental ranges but never enters either fitter.
    """
    if not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    noise_levels = tuple(noise_levels)
    if not noise_levels or len(set(noise_levels)) != len(noise_levels):
        raise ValueError("noise_levels must be nonempty and unique")
    if any(not np.isfinite(n) or n < 0 for n in noise_levels):
        raise ValueError("noise levels must be finite and nonnegative")
    # Validate shared inputs before constructing ranges or starting a long run.
    generate_data(K=K, q_max=q_max, samples=samples, noise=0, seed=seed)
    streams = iter(np.random.SeedSequence(seed).spawn(len(REGIMES) * len(noise_levels) * repeats))
    records = []
    for regime, (x_min, x_max) in REGIMES.items():
        for noise in noise_levels:
            for repeat in range(repeats):
                data = generate_data(K, q_max, x_min / K, x_max / K, samples, noise, next(streams))
                for fit in fit_both(data.concentration, data.observed, data.sigma):
                    records.append({
                        "regime": regime, "noise": noise, "repeat": repeat,
                        "samples": samples, "true_K": K, "true_q_max": q_max,
                        "c_min": float(data.concentration[0]), "c_max": float(data.concentration[-1]),
                        "sigma": data.sigma, **fit_record(fit, K, q_max),
                    })
    return records


def summarize(records):
    """Summarize valid-fit recovery while keeping failure rates visible."""
    groups = {}
    for row in records:
        key = (row["regime"], row["noise"], row["method"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (regime, noise, method), rows in groups.items():
        valid = [r for r in rows if r["status"] in ("ok", "warning")]
        summary = {
            "regime": regime, "noise": noise, "method": method,
            "attempts": len(rows), "valid_fits": len(valid),
            "failure_rate": 1 - len(valid) / len(rows),
            "warning_rate": sum(r["status"] == "warning" for r in rows) / len(rows),
            "median_runtime_seconds": float(np.median([r["runtime_seconds"] for r in rows])),
            "median_rmse": float(np.median([r["rmse"] for r in valid])) if valid else None,
        }
        for parameter in ("K", "q_max"):
            errors = [r[f"{parameter}_error_percent"] for r in valid]
            for label, value in (
                ("median_error_percent", np.median(errors) if errors else None),
                ("median_absolute_error_percent", np.median(np.abs(errors)) if errors else None),
                ("error_p10_percent", np.percentile(errors, 10) if errors else None),
                ("error_p90_percent", np.percentile(errors, 90) if errors else None),
            ):
                summary[f"{parameter}_{label}"] = float(value) if value is not None else None
        summaries.append(summary)
    return summaries
