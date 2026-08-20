import numpy as np
import pytest

from eq_toolkit.model.likelihood import (
    temporal_log_likelihood,
)


def test_likelihood_is_finite():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    magnitudes = np.array([3.0, 3.5, 4.0, 3.0])

    result = temporal_log_likelihood(
        times,
        magnitudes,
        mu=0.5,
        K=0.1,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
        t_start=0.0,
        t_end=3.0,
    )

    assert np.isfinite(result)


def test_likelihood_with_zero_productivity():
    times = np.array([0.0, 1.0, 2.0])
    magnitudes = np.array([3.0, 3.0, 3.0])

    result = temporal_log_likelihood(
        times,
        magnitudes,
        mu=0.5,
        K=0.0,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
        t_start=0.0,
        t_end=2.0,
    )

    # With K=0, the process is simply Poisson:
    # log L = N log(mu) - mu*T
    expected = (3 * np.log(0.5)- 0.5 * 2.0)

    assert np.isclose(result, expected)


def test_longer_observation_window_changes_likelihood():
    times = np.array([0.0, 1.0, 2.0])
    magnitudes = np.array([3.0, 3.0, 3.0])

    short = temporal_log_likelihood(
        times,
        magnitudes,
        mu=0.5,
        K=0.1,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
        t_start=0.0,
        t_end=2.0,
    )

    long = temporal_log_likelihood(
        times,
        magnitudes,
        mu=0.5,
        K=0.1,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
        t_start=0.0,
        t_end=3.0,
    )

    assert short != long


def test_invalid_p():
    times = np.array([0.0, 1.0])
    magnitudes = np.array([3.0, 3.0])

    with pytest.raises(ValueError):
        temporal_log_likelihood(
            times,
            magnitudes,
            mu=0.5,
            K=0.1,
            alpha=1.0,
            M0=3.0,
            c=0.1,
            p=1.0,
            t_start=0.0,
            t_end=1.0,
        )


def test_invalid_window():
    times = np.array([0.0, 1.0])
    magnitudes = np.array([3.0, 3.0])

    with pytest.raises(ValueError):
        temporal_log_likelihood(
            times,
            magnitudes,
            mu=0.5,
            K=0.1,
            alpha=1.0,
            M0=3.0,
            c=0.1,
            p=1.2,
            t_start=2.0,
            t_end=1.0,
        )


def test_mismatched_input_lengths():
    times = np.array([0.0, 1.0])
    magnitudes = np.array([3.0])

    with pytest.raises(ValueError):
        temporal_log_likelihood(
            times,
            magnitudes,
            mu=0.5,
            K=0.1,
            alpha=1.0,
            M0=3.0,
            c=0.1,
            p=1.2,
            t_start=0.0,
            t_end=1.0,
        )