"""Command-line entry points for single datasets and repeated experiments."""

import argparse
from pathlib import Path

from .reports import demo_report, sweep_report


def _seed(value):
    seed = int(value)
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be nonnegative")
    return seed


def build_parser():
    parser = argparse.ArgumentParser(description="Fit synthetic Langmuir adsorption measurements and study parameter recovery.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("demo", "sweep"):
        sub = commands.add_parser(command, help="Single dataset report" if command == "demo" else "Coverage and noise experiment")
        sub.add_argument("--K", type=float, default=2.0, help="True equilibrium constant (reciprocal concentration units)")
        sub.add_argument("--q-max", type=float, default=10.0, help="True saturation capacity")
        sub.add_argument("--samples", type=int, default=30, help="Concentrations per dataset (at least 3)")
        sub.add_argument("--seed", type=_seed, default=42, help="Nonnegative random seed")
        sub.add_argument("--output", type=Path, default=Path("artifacts") / command, help="Report directory; existing report files are replaced")
        if command == "demo":
            sub.add_argument("--noise", type=float, default=0.03, help="Measurement sigma as a fraction of q_max")
            sub.add_argument("--c-min", type=float, help="Lowest concentration (default: 0.01/K)")
            sub.add_argument("--c-max", type=float, help="Highest concentration (default: 100/K)")
        else:
            sub.add_argument("--noise-levels", nargs="+", type=float, default=[0, 0.01, 0.05, 0.1], help="Noise fractions of q_max")
            sub.add_argument("--repeats", type=int, default=100, help="Datasets per regime and noise level")
            sub.add_argument("--example-noise", type=float, default=0.03, help="Noise fraction for the separate coverage illustration")
    return parser


def _number(value):
    return f"{value:.6g}" if value is not None else "n/a"


def main(argv=None):
    parser = build_parser()
    options = vars(parser.parse_args(argv))
    command = options.pop("command")
    try:
        if command == "demo":
            K = options["K"]
            if K <= 0:
                raise ValueError("K must be finite and positive")
            if options["c_min"] is None:
                options["c_min"] = 0.01 / K
            if options["c_max"] is None:
                options["c_max"] = 100 / K
            records = demo_report(**options)
            print("method       status        K             q_max         RMSE          runtime (s)")
            for row in records:
                print(f"{row['method']:<12} {row['status']:<13} {_number(row['K']):<13} "
                      f"{_number(row['q_max']):<13} {_number(row['rmse']):<13} {_number(row['runtime_seconds'])}")
                print(f"  {row['message']}")
            exit_code = 0 if any(row["status"] in ("ok", "warning") for row in records) else 1
        else:
            summary = sweep_report(**options)
            print("regime       noise     method       valid/attempts  failures  median |K error|  median |q_max error|")
            for row in summary:
                print(f"{row['regime']:<12} {row['noise']:<9g} {row['method']:<12} "
                      f"{row['valid_fits']:>3}/{row['attempts']:<10} {row['failure_rate']:>7.1%}  "
                      f"{_number(row['K_median_absolute_error_percent']):>14}%  "
                      f"{_number(row['q_max_median_absolute_error_percent']):>18}%")
            exit_code = 0
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Reports: {options['output'].resolve()}")
    return exit_code
