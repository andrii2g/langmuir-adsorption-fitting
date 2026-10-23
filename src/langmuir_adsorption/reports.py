"""Write reproducible experiment inputs, raw results, summaries, and SVGs."""

import csv
import json
from pathlib import Path
import platform

import matplotlib
import numpy as np
import scipy

from .experiments import REGIMES, fit_record, run_sweep, summarize
from .fitting import fit_both
from .plotting import adsorption_plot, coverage_plot, recovery_plot, residual_plot
from .synthetic import generate_data


def write_csv(path, records):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_metadata(path, config):
    metadata = {"configuration": config, "versions": {
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
    }}
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def demo_report(output="artifacts/demo", **config):
    data = generate_data(**config)
    fits = fit_both(data.concentration, data.observed, data.sigma)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Store resolved values, including defaults, rather than only CLI overrides.
    resolved = {"K": data.K, "q_max": data.q_max, "c_min": float(data.concentration[0]),
                "c_max": float(data.concentration[-1]), "samples": len(data.concentration),
                "noise": data.sigma / data.q_max, "seed": config.get("seed", 42)}
    _write_metadata(output / "metadata.json", resolved)
    write_csv(output / "measurements.csv", [
        {"concentration": c, "observed": q, "true_quantity": truth}
        for c, q, truth in zip(data.concentration, data.observed, data.true_quantity)
    ])
    records = [fit_record(fit, data.K, data.q_max) for fit in fits]
    write_csv(output / "fits.csv", records)
    adsorption_plot(data, fits, output / "adsorption.svg")
    residual_plot(data, fits, output / "residuals.svg")
    return records


def sweep_report(output="artifacts/sweep", K=2.0, q_max=10.0, samples=30,
                 noise_levels=(0.0, 0.01, 0.05, 0.1), repeats=100, seed=42,
                 example_noise=0.03):
    config = dict(K=K, q_max=q_max, samples=samples, noise_levels=list(noise_levels), repeats=repeats, seed=seed)
    # Validate example settings before running the potentially expensive sweep.
    generate_data(K=K, q_max=q_max, samples=samples, noise=example_noise, seed=seed)
    records = run_sweep(**config)
    summary = summarize(records)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_metadata(output / "metadata.json", {**config, "example_noise": example_noise, "regimes_KC": REGIMES})
    write_csv(output / "fits.csv", records)
    write_csv(output / "summary.csv", summary)
    examples, example_rows, example_fits = {}, [], []
    streams = np.random.SeedSequence(seed).spawn(len(REGIMES))
    for (regime, bounds), stream in zip(REGIMES.items(), streams):
        data = generate_data(K, q_max, bounds[0] / K, bounds[1] / K, samples, example_noise, stream)
        fits = fit_both(data.concentration, data.observed, data.sigma)
        examples[regime] = (data, fits)
        example_rows.extend({"regime": regime, "concentration": c, "observed": q, "true_quantity": truth}
                            for c, q, truth in zip(data.concentration, data.observed, data.true_quantity))
        example_fits.extend({"regime": regime, **fit_record(fit, K, q_max)} for fit in fits)
    write_csv(output / "coverage_measurements.csv", example_rows)
    write_csv(output / "coverage_fits.csv", example_fits)
    coverage_plot(examples, output / "coverage.svg")
    recovery_plot(records, output / "parameter_recovery.svg")
    return summary
