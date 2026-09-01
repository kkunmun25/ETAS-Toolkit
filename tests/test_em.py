import numpy as np
import pytest

from eq_toolkit.calibrate.em import (EMResult,run_em,check_monotonicity,run_em_restarts)
from eq_toolkit.calibrate.mstep import ETASParameters

def make_synthetic_catalog():
    """Create a small deterministic catalog for EM testing."""

    times = np.array([
        0.0,
        0.5,
        1.0,
        1.8,
        2.5,
        3.2,
        4.0,
        5.0,
        6.0,
        7.5,
    ])

    magnitudes = np.array([
        2.0,
        2.4,
        2.8,
        2.2,
        3.0,
        2.5,
        2.1,
        2.7,
        2.3,
        2.6,
    ])

    return times, magnitudes


def make_parameters():
    return ETASParameters(
        mu=0.2,
        K=0.5,
        alpha=1.0,
        c=0.1,
        p=1.2,
    )


def test_em_result_valid():

    rho = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.2, 0.0],
        ]
    )

    bg = np.array([
        1.0,
        0.8,
        0.5,
    ])

    result = EMResult(
        parameters=make_parameters(),
        log_likelihood=-10.0,
        iterations=5,
        converged=True,
        rho=rho,
        bg=bg,
    )

    result.validate()


def test_em_result_rejects_invalid_probability_invariant():

    rho = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
        ]
    )

    bg = np.array([
        1.0,
        0.8,
    ])

    result = EMResult(
        parameters=make_parameters(),
        log_likelihood=-10.0,
        iterations=5,
        converged=True,
        rho=rho,
        bg=bg,
    )

    with pytest.raises(
        ValueError,
        match="probability invariant",
    ):
        result.validate()


def test_em_result_rejects_negative_rho():

    rho = np.array(
        [
            [0.0, 0.0],
            [-0.2, 0.0],
        ]
    )

    bg = np.array([
        1.0,
        1.0,
    ])

    result = EMResult(
        parameters=make_parameters(),
        log_likelihood=-10.0,
        iterations=5,
        converged=True,
        rho=rho,
        bg=bg,
    )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        result.validate()

def test_run_em_returns_result():

    times = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
        2.2,
        2.8,
    ])

    initial = ETASParameters(
        mu=0.2,
        K=0.5,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    result = run_em(
        times,
        magnitudes,
        initial,
        max_iterations=10,
    )

    assert isinstance(result, EMResult)

    assert result.iterations >= 1

    assert np.isfinite(
        result.log_likelihood
    )

    result.validate()


def test_run_em_preserves_probability_invariant():

    times = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
        2.2,
    ])

    initial = ETASParameters(
        mu=0.2,
        K=0.3,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    result = run_em(
        times,
        magnitudes,
        initial,
        max_iterations=5,
    )

    probabilities = (
        result.bg
        + result.rho.sum(axis=1)
    )

    assert np.allclose(
        probabilities,
        1.0,
        atol=1e-10,
    )


def test_run_em_rejects_too_few_events():

    times = np.array([0.0])
    magnitudes = np.array([2.0])

    initial = ETASParameters(
        mu=0.2,
        K=0.3,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    with pytest.raises(
        ValueError,
        match="two events",
    ):
        run_em(
            times,
            magnitudes,
            initial,
        ) 

def test_check_monotonicity_accepts_increase():

    check_monotonicity(
        -10.0,
        -9.0,
    )


def test_check_monotonicity_accepts_tiny_difference():

    check_monotonicity(
        -10.0,
        -10.00000000001,
        tolerance=1e-10,
    )


def test_check_monotonicity_rejects_decrease():

    with pytest.raises(
        ValueError,
        match="decreased",
    ):
        check_monotonicity(
            -10.0,
            -12.0,
        )


def test_check_monotonicity_rejects_nonfinite():

    with pytest.raises(
        ValueError,
        match="previous_q",
    ):
        check_monotonicity(
            np.inf,
            -10.0,
        ) 

def test_run_em_restarts_returns_best_result():

    times = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
        2.2,
        2.8,
    ])

    initial_parameters = [
        ETASParameters(
            mu=0.2,
            K=0.3,
            alpha=0.5,
            c=0.1,
            p=1.2,
        ),
        ETASParameters(
            mu=0.3,
            K=0.5,
            alpha=0.8,
            c=0.2,
            p=1.5,
        ),
    ]

    result = run_em_restarts(
        times,
        magnitudes,
        initial_parameters,
        max_iterations=5,
    )

    assert isinstance(result, EMResult)
    assert np.isfinite(result.log_likelihood)
    result.validate()


def test_run_em_restarts_requires_initial_parameters():

    times = np.array([0.0, 1.0])
    magnitudes = np.array([2.0, 2.5])

    with pytest.raises(
        ValueError,
        match="one initial",
    ):
        run_em_restarts(
            times,
            magnitudes,
            [],
        )


def test_run_em_restarts_single_start():

    times = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
        2.2,
    ])

    initial = ETASParameters(
        mu=0.2,
        K=0.3,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    result = run_em_restarts(
        times,
        magnitudes,
        [initial],
        max_iterations=5,
    )

    assert isinstance(result, EMResult)

def test_em_end_to_end_synthetic_catalog():

    times, magnitudes = make_synthetic_catalog()

    initial_parameters = [
        ETASParameters(
            mu=0.2,
            K=0.3,
            alpha=0.5,
            c=0.1,
            p=1.2,
        ),
        ETASParameters(
            mu=0.4,
            K=0.2,
            alpha=0.8,
            c=0.2,
            p=1.5,
        ),
    ]

    result = run_em_restarts(
        times,
        magnitudes,
        initial_parameters,
        max_iterations=20,
        tolerance=1e-5,
    )

    assert isinstance(result, EMResult)

    # EM must return a finite objective.
    assert np.isfinite(result.log_likelihood)

    # Parameters must be valid.
    result.parameters.validate()

    # E-step probability invariant.
    probabilities = (
        result.bg
        + result.rho.sum(axis=1)
    )

    assert np.allclose(
        probabilities,
        1.0,
        atol=1e-10,
    )

    # The result must contain all events.
    assert result.rho.shape == (
        len(times),
        len(times),
    )

    assert result.bg.shape == (
        len(times),
    )                    


def test_run_em_respects_max_iterations():

    times = np.array([0.0, 1.0, 2.0, 3.0])
    magnitudes = np.array([2.0, 2.5, 3.0, 2.2])

    initial = ETASParameters(
        mu=0.2,
        K=0.3,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    result = run_em(
        times,
        magnitudes,
        initial,
        max_iterations=2,
    )

    assert result.iterations <= 2


def test_run_em_reports_valid_convergence_state():

    times = np.array([0.0, 1.0, 2.0, 3.0])
    magnitudes = np.array([2.0, 2.5, 3.0, 2.2])

    initial = ETASParameters(
        mu=0.2,
        K=0.3,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    result = run_em(
        times,
        magnitudes,
        initial,
        max_iterations=10,
    )

    assert isinstance(result.converged, (bool, np.bool_))
    assert result.iterations >= 1
    assert np.isfinite(result.log_likelihood)    