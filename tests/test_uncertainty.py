import numpy as np
import pytest

from eq_toolkit.calibrate.mstep import ETASParameters
from eq_toolkit.calibrate.uncertainty import (
    check_compensator_bug,
    compute_hessian_uncertainty,
    parametric_bootstrap_uncertainty,
)
from eq_toolkit.model.simulate import simulate_etas


def test_check_compensator_bug_detects_bug():
    # Verify that the bug detection correctly identifies the missing compensator
    assert check_compensator_bug() is True


def test_simulate_etas_basic():
    times, mags = simulate_etas(
        mu=0.5,
        K=0.05,
        alpha=1.0,
        c=0.05,
        p=1.2,
        m0=2.5,
        duration=100.0,
        beta=2.3,
        seed=123,
    )
    assert len(times) > 0
    assert len(times) == len(mags)
    assert np.all(np.diff(times) >= 0)  # Chronologically sorted
    assert np.all(mags >= 2.5)
    assert times[-1] <= 100.0


def test_simulate_etas_validation():
    with pytest.raises(ValueError):
        simulate_etas(mu=-0.1, K=0.1, alpha=1.0, c=0.05, p=1.2)
    with pytest.raises(ValueError):
        simulate_etas(mu=0.5, K=0.1, alpha=1.0, c=0.05, p=1.0)  # p must be > 1


def test_compute_hessian_uncertainty_structure():
    # Small test sequence
    times = np.linspace(0.0, 50.0, 20)
    mags = np.full(20, 2.8)
    params = ETASParameters(mu=0.4, K=0.02, alpha=1.0, c=0.05, p=1.3)

    res = compute_hessian_uncertainty(
        times=times,
        magnitudes=mags,
        parameters=params,
        m0=2.5,
        likelihood_func="likelihood.py",
        step_scale=1e-3,
    )

    assert res.hessian.shape == (5, 5)
    assert res.fisher_info.shape == (5, 5)
    assert len(res.standard_errors) == 5
    assert len(res.ci_95_lower) == 5
    assert len(res.ci_95_upper) == 5
    assert res.correlation_matrix.shape == (5, 5)
    # Symmetry of Hessian
    assert np.allclose(res.hessian, res.hessian.T, atol=1e-5)


def test_bootstrap_uncertainty_structure():
    params = ETASParameters(mu=0.5, K=0.03, alpha=1.1, c=0.04, p=1.4)
    res = parametric_bootstrap_uncertainty(
        parameters=params,
        m0=2.5,
        duration=50.0,
        beta=2.3,
        n_bootstraps=3,
        max_iterations=10,
        tolerance=1e-3,
        verbose=False,
    )
    assert res.n_requested == 3
    assert 0.0 <= res.exclusion_rate <= 1.0
    assert len(res.bootstrap_se) == 5
