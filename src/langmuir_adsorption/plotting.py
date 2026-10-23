"""Headless SVG figures in the original adsorption measurement space."""

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

from .experiments import REGIMES
from .model import langmuir


COLORS = {"nonlinear": "#d55e00", "linearized": "#009e73"}


def _save(fig, path):
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def _curves(ax, data, fits, grid):
    ax.plot(grid, langmuir(grid, data.K, data.q_max), color="#333333", label="True model", lw=2)
    ax.scatter(data.concentration, data.observed, color="#0072b2", s=22, label="Measurements", zorder=3)
    for fit in fits:
        if fit.valid:
            ax.plot(grid, fit.predict(grid), color=COLORS[fit.method], ls="--",
                    label=f"{fit.method}: K={fit.K:.3g}, q_max={fit.q_max:.3g}")
        else:
            ax.plot([], [], color=COLORS[fit.method], ls="--", label=f"{fit.method}: {fit.status}")
    ax.set(xscale="log", xlabel="Concentration C", ylabel="Adsorbed quantity q")
    ax.grid(alpha=0.2)


def adsorption_plot(data, fits, path):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    grid = np.geomspace(data.concentration[0], data.concentration[-1], 400)
    _curves(ax, data, fits, grid)
    ax.set_title(f"Langmuir adsorption: true K={data.K:g}, q_max={data.q_max:g}, sigma={data.sigma:g}")
    ax.legend(fontsize=9)
    _save(fig, path)


def residual_plot(data, fits, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for fit in fits:
        if fit.valid:
            residual = data.observed - fit.predict(data.concentration)
            label = f"{fit.method} (RMSE={fit.rmse:.3g})"
            axes[0].scatter(data.concentration, residual, color=COLORS[fit.method], label=label, alpha=0.8)
            axes[1].hist(residual, bins="auto", color=COLORS[fit.method], alpha=0.45, label=fit.method)
    axes[0].axhline(0, color="#333333", lw=1)
    axes[0].set(xscale="log", xlabel="Concentration C", ylabel="Residual: observed - fitted q")
    axes[1].set(xlabel="Residual", ylabel="Count")
    for ax in axes:
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Residual diagnostics in original q units")
    _save(fig, path)


def coverage_plot(examples, path):
    """examples maps regime names to (Dataset, list[FitResult])."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for ax, (regime, (data, fits)) in zip(axes.flat, examples.items()):
        grid = np.geomspace(0.001 / data.K, 100 / data.K, 500)
        _curves(ax, data, fits, grid)
        ax.axvspan(data.concentration[0], data.concentration[-1], color="#0072b2", alpha=0.08)
        ax.axvline(1 / data.K, color="#777777", lw=1, ls=":")
        ax.set_ylim(min(-0.1 * data.q_max, min(data.observed) * 1.2),
                    max(1.25 * data.q_max, max(data.observed) * 1.1))
        ax.set_title(f"{regime}: KC = {REGIMES[regime][0]:g} to {REGIMES[regime][1]:g}")
        ax.legend(fontsize=8)
    fig.suptitle("Coverage and extrapolation: shaded region is measured; dotted line is C=1/K")
    _save(fig, path)


def recovery_plot(records, path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained", sharex=True)
    for column, regime in enumerate(REGIMES):
        for row, parameter in enumerate(("K", "q_max")):
            ax = axes[row, column]
            for method, color in COLORS.items():
                selected = [r for r in records if r["regime"] == regime and r["method"] == method]
                levels = sorted({r["noise"] for r in selected})
                median, low, high = [], [], []
                for noise in levels:
                    errors = [abs(r[f"{parameter}_error_percent"]) for r in selected
                              if r["noise"] == noise and r["status"] in ("ok", "warning")]
                    quantiles = np.percentile(errors, [10, 50, 90]) if errors else [np.nan] * 3
                    low.append(quantiles[0])
                    median.append(quantiles[1])
                    high.append(quantiles[2])
                x = np.array(levels) * 100
                ax.plot(x, median, marker="o", color=color, label=method)
                ax.fill_between(x, low, high, color=color, alpha=0.15)
            ax.set(title=f"{regime}: {parameter}", xlabel="Noise sigma / q_max (%)",
                   ylabel="Absolute parameter error (%)", yscale="symlog")
            ax.grid(alpha=0.2)
            if row == 0 and column == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Parameter recovery: median and 10th-90th percentiles of valid fits; see summary.csv for failure rates")
    _save(fig, path)
