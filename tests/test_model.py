import numpy as np
import pytest

from langmuir_adsorption import coverage, generate_data, langmuir


def test_physical_limits_and_half_saturation():
    assert coverage(0, 2) == 0
    assert coverage(0.5, 2) == 0.5
    assert langmuir(0.5, 2, 10) == 5
    assert coverage(1e12, 2) == pytest.approx(1)
    assert coverage(1e308, 1e308) == 1
    assert np.all(np.diff(langmuir(np.geomspace(1e-8, 1e8, 100), 2, 10)) > 0)


@pytest.mark.parametrize("concentration,K,q_max", [([-1], 2, 10), ([np.nan], 2, 10), ([1], 0, 10), ([1], 2, -1)])
def test_invalid_model_inputs(concentration, K, q_max):
    with pytest.raises(ValueError):
        langmuir(concentration, K, q_max)


def test_seeded_noise_is_reproducible_and_not_clipped():
    first = generate_data(c_min=1e-8, c_max=1e-6, noise=0.1)
    second = generate_data(c_min=1e-8, c_max=1e-6, noise=0.1)
    np.testing.assert_array_equal(first.observed, second.observed)
    assert np.any(first.observed < 0)
    assert first.sigma == 1


def test_noiseless_data_and_log_spacing():
    data = generate_data(noise=0)
    np.testing.assert_array_equal(data.observed, data.true_quantity)
    np.testing.assert_allclose(np.diff(np.log(data.concentration)), np.log(1e4) / 29)


@pytest.mark.parametrize("kwargs", [{"c_min": 0}, {"c_max": 0.001}, {"samples": 2}, {"noise": -1}, {"noise": np.nan}])
def test_invalid_generation_inputs(kwargs):
    with pytest.raises(ValueError):
        generate_data(**kwargs)
