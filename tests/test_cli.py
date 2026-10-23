import pytest

from langmuir_adsorption.cli import main


def test_demo_cli_generates_report_and_prints_estimates(tmp_path, capsys):
    assert main(["demo", "--noise", "0", "--output", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "nonlinear" in output and "linearized" in output
    assert "Reports:" in output
    assert (tmp_path / "fits.csv").is_file()


@pytest.mark.parametrize("arguments", [
    ["demo", "--K", "0"], ["demo", "--K", "nan"],
    ["demo", "--noise", "-0.1"], ["demo", "--samples", "2"],
    ["sweep", "--repeats", "0"], ["sweep", "--noise-levels", "-1"],
    ["demo", "--seed", "-1"],
])
def test_invalid_cli_configuration_exits_cleanly(arguments, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main([*arguments, "--output", str(tmp_path / "invalid")])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "invalid").exists()
