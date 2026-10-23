import numpy as np
import pytest

from langmuir_adsorption import generate_data, langmuir
from langmuir_adsorption.fitting import fit_both, fit_linearized, fit_nonlinear


@pytest.mark.parametrize("K,q_max", [(2, 10), (0.001, 10000), (1000, 0.01)])
def test_noiseless_parameter_recovery_across_units(K, q_max):
    data = generate_data(K=K, q_max=q_max, c_min=0.01 / K, c_max=100 / K, noise=0)
    for fit in fit_both(data.concentration, data.observed, data.sigma):
        assert fit.valid, fit.message
        assert fit.K == pytest.approx(K, rel=1e-5)
        assert fit.q_max == pytest.approx(q_max, rel=1e-5)
        assert fit.rmse < q_max * 1e-7


def test_noisy_fit_metrics_use_original_observations():
    data = generate_data(noise=0.01)
    fits = fit_both(data.concentration, data.observed, data.sigma)
    for fit in fits:
        assert fit.valid
        residual = data.observed - fit.predict(data.concentration)
        assert fit.rmse == pytest.approx(np.sqrt(np.mean(residual**2)))
        assert fit.runtime_seconds > 0
    assert fits[0].rmse <= fits[1].rmse + 1e-8
    assert fits[0].K_std > 0


def test_zero_denominator_is_an_explicit_failure():
    result = fit_linearized([1, 2, 3], [1, 0, 2])
    assert result.status == "failed"
    with pytest.raises(ValueError):
        result.predict([1])


def test_unphysical_linear_estimate_is_retained_but_flagged():
    result = fit_linearized([1, 2, 3], [1, 2, 4])
    assert result.status == "unphysical"
    assert result.K < 0
    assert result.rmse is None


def test_negative_measurements_are_not_dropped():
    c, q = np.array([1, 2, 3, 4.]), np.array([-0.1, 1, 2, 3.])
    a, b = np.linalg.lstsq(np.column_stack((np.ones(4), c)), c / q, rcond=None)[0]
    result = fit_linearized(c, q)
    assert result.K == pytest.approx(b / a)
    assert result.q_max == pytest.approx(1 / b)


def test_low_concentration_identifies_product_but_not_individual_parameters():
    c = np.geomspace(1e-6, 1e-3, 30)
    np.testing.assert_allclose(langmuir(c, 1, 10), langmuir(c, 0.5, 20), rtol=0.001)
    result = fit_nonlinear(c, langmuir(c, 1, 10), sigma=0.01)
    assert result.status == "warning"
    assert result.parameter_correlation < -0.99


@pytest.mark.parametrize("c,q", [([1, 2], [1, 2]), ([1, 1, 1], [1, 2, 3]), ([1, 2, 3], [1, np.nan, 2])])
def test_invalid_observations(c, q):
    with pytest.raises(ValueError):
        fit_both(c, q)
