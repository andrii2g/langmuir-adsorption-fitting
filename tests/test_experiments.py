import json
from xml.etree import ElementTree

import pytest

from langmuir_adsorption.experiments import run_sweep, summarize
from langmuir_adsorption.reports import demo_report, sweep_report


def test_paired_sweep_is_reproducible_except_runtime():
    first = run_sweep(noise_levels=(0, 0.05), repeats=2)
    second = run_sweep(noise_levels=(0, 0.05), repeats=2)
    assert len(first) == 4 * 2 * 2 * 2
    for a, b in zip(first, second):
        assert {k: v for k, v in a.items() if k != "runtime_seconds"} == {
            k: v for k, v in b.items() if k != "runtime_seconds"}
    for a, b in zip(first[::2], first[1::2]):
        assert (a["regime"], a["noise"], a["repeat"]) == (b["regime"], b["noise"], b["repeat"])
        if a["rmse"] is not None and b["rmse"] is not None:
            assert a["rmse"] <= b["rmse"] + 1e-6


def test_summary_preserves_failures_without_inventing_errors():
    rows = [dict(regime="low", noise=0.1, method="linearized", status="failed", runtime_seconds=0.1)]
    summary = summarize(rows)[0]
    assert summary["failure_rate"] == 1
    assert summary["valid_fits"] == 0
    assert summary["K_median_absolute_error_percent"] is None
    assert summary["median_rmse"] is None


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"noise_levels": ()}, {"noise_levels": (0.1, 0.1)}, {"noise_levels": (-1,)}])
def test_invalid_sweep_configuration(kwargs):
    with pytest.raises(ValueError):
        run_sweep(**kwargs)


def test_reports_write_metrics_metadata_and_valid_svg(tmp_path):
    demo_report(tmp_path / "demo", noise=0)
    sweep_report(tmp_path / "sweep", repeats=1, noise_levels=(0, 0.05))
    for directory, names in (("demo", ("adsorption", "residuals")),
                             ("sweep", ("coverage", "parameter_recovery"))):
        metadata = json.loads((tmp_path / directory / "metadata.json").read_text())
        assert metadata["configuration"]["seed"] == 42
        assert "scipy" in metadata["versions"]
        for name in names:
            svg = ElementTree.parse(tmp_path / directory / f"{name}.svg").getroot()
            assert svg.tag.endswith("svg")
    assert (tmp_path / "sweep" / "summary.csv").is_file()
