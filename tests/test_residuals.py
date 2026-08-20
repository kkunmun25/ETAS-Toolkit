import numpy as np
import pytest

from eq_toolkit.model.residuals import (
    transformed_time_residuals,
    ks_test_residuals,
)


def test_residuals_have_correct_length():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    magnitudes = np.array([3.0, 3.0, 3.5, 3.0])

    residuals = transformed_time_residuals(
        times,
        magnitudes,
        mu=0.5,
        K=0.1,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
    )

    assert len(residuals) == len(times) - 1


def test_residuals_are_nonnegative():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    magnitudes = np.array([3.0, 3.0, 3.5, 3.0])

    residuals = transformed_time_residuals(
        times,
        magnitudes,
        mu=0.5,
        K=0.1,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
    )

    assert np.all(residuals >= 0)


def test_poisson_case():
    """
    With K=0, the ETAS model becomes a homogeneous
    Poisson process.

    Therefore the transformed residual is simply
    mu * delta_t.
    """

    times = np.array([0.0, 1.0, 3.0])
    magnitudes = np.array([3.0, 3.0, 3.0])

    residuals = transformed_time_residuals(
        times,
        magnitudes,
        mu=2.0,
        K=0.0,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
    )

    expected = np.array([
        2.0,
        4.0,
    ])

    assert np.allclose(
        residuals,
        expected,
    )


def test_unsorted_times_raise_error():
    times = np.array([0.0, 2.0, 1.0])
    magnitudes = np.array([3.0, 3.0, 3.0])

    with pytest.raises(ValueError):
        transformed_time_residuals(
            times,
            magnitudes,
            mu=0.5,
            K=0.1,
            alpha=1.0,
            M0=3.0,
            c=0.1,
            p=1.2,
        )


def test_single_event_raises_error():
    times = np.array([0.0])
    magnitudes = np.array([3.0])

    with pytest.raises(ValueError):
        transformed_time_residuals(
            times,
            magnitudes,
            mu=0.5,
            K=0.1,
            alpha=1.0,
            M0=3.0,
            c=0.1,
            p=1.2,
        )


def test_ks_test_returns_valid_values():
    residuals = np.array([
        0.2,
        0.5,
        0.8,
        1.1,
        1.5,
        2.0,
    ])

    statistic, p_value = ks_test_residuals(
        residuals
    )

    assert 0.0 <= statistic <= 1.0
    assert 0.0 <= p_value <= 1.0


def test_ks_test_rejects_negative_residuals():
    residuals = np.array([
        0.2,
        -0.5,
        1.0,
    ])

    with pytest.raises(ValueError):
        ks_test_residuals(residuals)