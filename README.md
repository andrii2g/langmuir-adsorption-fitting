# Langmuir adsorption fitting

A small Python experiment in surface chemistry and parameter estimation. Generate
synthetic equilibrium adsorption measurements, fit the Langmuir isotherm using
nonlinear and linearized least squares, and investigate how concentration coverage
and measurement noise affect the recovered parameters.

The central lesson: **a small prediction error does not guarantee accurate physical
parameters**. When measurements cover only the initial rise, many combinations of
affinity and saturation capacity describe almost the same curve.

## Install

Requires Python 3.10 or newer. From the repository root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

On Linux/macOS use `.venv/bin/python` in place of `.\.venv\Scripts\python`.
The runtime dependencies are NumPy, SciPy, and Matplotlib; pytest is for development.
No GitHub Actions or workflows are included.

## Run a single experiment

```powershell
.\.venv\Scripts\python -m langmuir_adsorption demo --K 2 --q-max 10 --noise 0.03 --seed 42
```

The default dataset has 30 logarithmically spaced concentrations from `0.01/K` to
`100/K`. Noise `0.03` means an observation standard deviation of `0.03*q_max`,
not 3% of each individual observation. Concentration has no measurement error.

Override the range to demonstrate incomplete coverage:

```powershell
.\.venv\Scripts\python -m langmuir_adsorption demo --K 2 --q-max 10 --c-min 0.0005 --c-max 0.05 --samples 30 --noise 0.03 --output artifacts/low-coverage
```

The installed `langmuir` command provides the same interface. Use `demo --help` or
`sweep --help` for all options. Existing files with report names in the output
directory are replaced; use separate directories to preserve runs.

## Run the coverage/noise sweep

```powershell
.\.venv\Scripts\python -m langmuir_adsorption sweep --repeats 100 --noise-levels 0 0.01 0.05 0.1 --seed 42
```

This fits 1,600 datasets with both methods (3,200 fits). A separate random stream
generates each dataset, and both methods receive identical observations. Repeated
zero-noise datasets are identical and provide a numerical baseline.

| Regime | Dimensionless range KC | What it measures |
| --- | --- | --- |
| Low | 0.001–0.1 | Initial slope, approximately q_max*K |
| Transition | 0.1–10 | Bend around half-saturation |
| Broad | 0.01–100 | Initial rise, bend, and plateau |
| Saturation | 10–100 | Plateau; weaker information about K |

The generating K sets these controlled synthetic ranges. Neither fitter receives
the true K or q_max. Every dataset uses the same number of concentrations; changing
the range therefore also changes the spacing between measurements.

## Reports

Reports default to ignored directories under `artifacts/`.

For the default seeded demo (true K=2, q_max=10, noise=0.03), nonlinear fitting
recovers approximately K=2.010, q_max=10.036, and RMSE=0.228. Linearized fitting
gives K=1.925, q_max=10.119, and RMSE=0.233.

In the default 100-repeat sweep at 5% noise, nonlinear median absolute K error is
about 6.3% with broad coverage, versus 100% with low coverage. These are synthetic
results for this seed and noise model, not guarantees for real measurements.

| Command | File | Contents |
| --- | --- | --- |
| demo | `adsorption.svg` | Observations, true model, and both fitted curves |
| demo | `residuals.svg` | Residuals versus concentration and residual histograms |
| demo | `measurements.csv` | Concentrations, observations, and noiseless quantities |
| demo | `fits.csv` | Parameters, errors, RMSE, residual statistics, diagnostics, runtime |
| sweep | `coverage.svg` | Four ranges, measured-region shading, and extrapolated fits |
| sweep | `parameter_recovery.svg` | Median absolute parameter errors and empirical 10th–90th percentile bands |
| sweep | `fits.csv` | Every fit attempt, including failures and unphysical estimates |
| sweep | `summary.csv` | Recovery statistics, failure/warning rates, and median fit runtime |
| sweep | `coverage_measurements.csv`, `coverage_fits.csv` | Data and fits used in the separate coverage illustration |
| both | `metadata.json` | Resolved configuration, seed, and numerical-library versions |

Coverage illustrations use `--example-noise 0.03` by default and independent streams
from the repeated sweep. Parameter-error bands summarize **valid fits only**; consult
failure rates alongside them to avoid a misleading impression of reliability.

Signed parameter error is `100*(estimate/true_value - 1)`. Summaries include median
signed error, median absolute error, and the 10th/90th percentiles of signed error.
RMSE and residuals (`observed - fitted`) always use the original q units, including
for the linearized fit. Residual standard deviation uses `ddof=1`. Runtime measures
each fitting call after input validation, excluding simulation, plotting, and file
output; it varies between runs and should not be treated as a precise benchmark.

## Physical model

The Langmuir isotherm assumes a finite set of equivalent, independent adsorption
sites with single occupancy at equilibrium:

```text
theta(C) = K*C / (1 + K*C)
q(C)     = q_max * theta(C)
```

Use consistent units: if C is in mg/L, K is in L/mg; q and q_max might be in mg/g.
The software does not convert units. It is an equilibrium model, not a kinetic model.
Half-saturation occurs at `C = 1/K`.

Synthetic observations add independent Gaussian noise with constant standard
deviation `sigma = noise*q_max`. Negative observations are retained rather than
clipped, preserving the specified statistical model.

### Nonlinear fitting

SciPy `curve_fit` minimizes squared residuals directly in q space, with positive
parameter bounds. Data-derived scaling and starting values avoid requiring the true
parameters. Known positive sigma is passed with `absolute_sigma=True`; zero-noise
experiments use unweighted fitting.

Returned parameter standard errors and correlation use the local covariance
approximation. A fit receives a `warning` when covariance is unavailable, relative
standard errors exceed one, absolute parameter correlation exceeds 0.99, or SciPy
warns about covariance. These are diagnostic heuristics, not guarantees of accuracy
or formal confidence intervals. The repeated experiments provide empirical recovery
distributions; they are not confidence intervals for a single observed dataset.

### Linearized fitting

The comparison method uses ordinary least squares on:

```text
C/q = a + b*C
q_max = 1/b
K = b/a
```

Although algebraically equivalent to the isotherm, this transformation changes the
error distribution and can amplify near-zero observations. It does not minimize
squared q residuals. Exact zero observations produce an explicit failed fit; finite
negative measurements are included. Nonpositive or nonfinite parameter estimates
are marked `unphysical`, excluded from recovery statistics, and never plotted as
physical curves. Finite unphysical parameter estimates remain in the raw CSV.

At low concentration, `q ≈ q_max*K*C`: the product can be well constrained while
the individual parameters remain ambiguous. Near saturation, `q ≈ q_max`, leaving
comparatively little information about K. Neither a successful optimizer exit nor
a low RMSE resolves missing experimental information.

## Python API

```python
from langmuir_adsorption import generate_data
from langmuir_adsorption.fitting import fit_both

data = generate_data(K=2, q_max=10, c_min=0.005, c_max=50, noise=0.03, seed=42)
for fit in fit_both(data.concentration, data.observed, sigma=data.sigma):
    print(fit.method, fit.status, fit.K, fit.q_max, fit.rmse)
```

`generate_data` uses fixed concentration defaults of 0.005–50; the CLI derives its
defaults from the requested K. `FitResult.valid` includes `ok` and `warning` fits.
Calling `predict` on a failed or unphysical fit raises `ValueError`. The CLI returns
2 for invalid inputs, 1 when neither demo fit is valid, and 0 for completed reports.
Sweep failures are experimental outcomes recorded in reports, not command failures.

## Development and verification

```powershell
.\.venv\Scripts\python -m pytest -q
```

Tests cover physical limits, reproducible unclipped noise, noiseless recovery across
unit scales, metrics in original units, weak parameter constraints, explicit invalid
fits, paired sweeps, report generation, and CLI input validation.

The MVP deliberately uses synthetic data, constant-variance quantity noise, and
ordinary least squares. Errors in concentration, alternative isotherms, experimental
CSV import, bootstrap intervals, and interactive interfaces are outside this scope.

## References

- [SciPy curve_fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
- [USDA: comparison of linear and nonlinear Langmuir fitting](https://www.ars.usda.gov/research/publications/publication/?seqNo115=197174)
