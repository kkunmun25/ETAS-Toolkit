import numpy as np
import pytest

from eq_toolkit.calibrate.mstep import (
    ETASParameters,
    expected_complete_log_likelihood,
    update_alpha,
    update_mu,
    update_K,
    update_p,
    update_c
)


def test_etas_parameters_valid():
    params = ETASParameters(
        mu=0.2,
        K=0.5,
        alpha=1.0,
        c=0.01,
        p=1.1,
    )

    params.validate()


def test_update_mu():
    bg = np.array([1.0, 0.8, 0.4, 0.2])

    mu = update_mu(
        bg,
        duration=10.0,
    )

    assert np.isclose(mu, 0.24)


def test_update_mu_rejects_negative_background_probability():
    bg = np.array([1.0, -0.1, 0.5])

    with pytest.raises(ValueError, match="negative"):
        update_mu(
            bg,
            duration=10.0,
        )


def test_update_mu_rejects_zero_duration():
    bg = np.array([1.0, 0.5])

    with pytest.raises(ValueError, match="positive"):
        update_mu(
            bg,
            duration=0.0,
        )


def test_etas_parameters_reject_invalid_mu():
    params = ETASParameters(
        mu=0.0,
        K=0.5,
        alpha=1.0,
        c=0.01,
        p=1.1,
    )

    with pytest.raises(ValueError, match="mu"):
        params.validate()


def test_update_K_returns_positive_value():
    times = np.array([0.0, 1.0, 2.0, 4.0])

    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
        2.2,
    ])

    rho = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.1, 0.3, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.0],
        ]
    )

    K = update_K(
        rho,
        times,
        magnitudes,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    assert np.isfinite(K)
    assert K > 0


def test_update_K_zero_triggering():
    times = np.array([0.0, 1.0, 2.0])

    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
    ])

    rho = np.zeros((3, 3))

    K = update_K(
        rho,
        times,
        magnitudes,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    assert K == 0.0


def test_update_K_rejects_wrong_rho_shape():
    times = np.array([0.0, 1.0, 2.0])

    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
    ])

    rho = np.zeros((2, 2))

    with pytest.raises(ValueError, match="shape"):
        update_K(
            rho,
            times,
            magnitudes,
            alpha=0.5,
            c=0.1,
            p=1.2,
        )

def test_update_alpha_returns_valid_value():
    times = np.array([0.0, 1.0, 2.0, 4.0])

    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
        2.2,
    ])

    rho = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.1, 0.3, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.0],
        ]
    )

    alpha = update_alpha(
        rho,
        magnitudes,
        m0=0.0,
    )

    assert np.isfinite(alpha)
    assert 0.0 <= alpha <= 5.0


def test_update_alpha_zero_triggering():
    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
    ])

    rho = np.zeros((3, 3))

    alpha = update_alpha(
        rho,
        magnitudes,
    )

    assert alpha == 0.0


def test_update_alpha_rejects_wrong_shape():
    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
    ])

    rho = np.zeros((2, 2))

    with pytest.raises(ValueError, match="shape"):
        update_alpha(
            rho,
            magnitudes,
        )


def test_update_alpha_respects_bounds():
    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
    ])

    rho = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.2, 0.4, 0.0],
        ]
    )

    alpha = update_alpha(
        rho,
        magnitudes,
        alpha_bounds=(0.0, 2.0),
    )

    assert 0.0 <= alpha <= 2.0

def test_update_c_returns_valid_value():
    times = np.array([0.0, 1.0, 2.0, 4.0])

    rho = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.1, 0.3, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.0],
        ]
    )

    c = update_c(
        rho,
        times,
        p=1.2,
    )

    assert np.isfinite(c)
    assert 1e-4 <= c <= 10.0


def test_update_c_zero_triggering():
    times = np.array([0.0, 1.0, 2.0])

    rho = np.zeros((3, 3))

    c = update_c(
        rho,
        times,
        p=1.2,
    )

    assert c == 1e-4


def test_update_p_returns_valid_value():
    times = np.array([0.0, 1.0, 2.0, 4.0])

    rho = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.1, 0.3, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.0],
        ]
    )

    p = update_p(
        rho,
        times,
        c=0.1,
    )

    assert np.isfinite(p)
    assert 0.5 <= p <= 3.0


def test_update_p_zero_triggering():
    times = np.array([0.0, 1.0, 2.0])

    rho = np.zeros((3, 3))

    p = update_p(
        rho,
        times,
        c=0.1,
    )

    assert p == 0.5


def test_expected_complete_log_likelihood_is_finite():
    times = np.array([0.0, 1.0, 2.0, 4.0])

    magnitudes = np.array([
        2.0,
        2.5,
        3.0,
        2.2,
    ])

    rho = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.1, 0.3, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.0],
        ]
    )

    bg = 1.0 - rho.sum(axis=1)

    value = expected_complete_log_likelihood(
        times,
        magnitudes,
        rho,
        bg,
        mu=0.5,
        K=0.2,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    assert np.isfinite(value)


def test_expected_complete_log_likelihood_rejects_invalid_c():
    times = np.array([0.0, 1.0])
    magnitudes = np.array([2.0, 2.5])

    rho = np.array([
        [0.0, 0.0],
        [0.2, 0.0],
    ])

    bg = np.array([1.0, 0.8])

    with pytest.raises(ValueError, match="c"):
        expected_complete_log_likelihood(
            times,
            magnitudes,
            rho,
            bg,
            mu=0.5,
            K=0.2,
            alpha=0.5,
            c=0.0,
            p=1.2,
        )


def test_expected_complete_log_likelihood_rejects_invalid_p():
    times = np.array([0.0, 1.0])
    magnitudes = np.array([2.0, 2.5])

    rho = np.array([
        [0.0, 0.0],
        [0.2, 0.0],
    ])

    bg = np.array([1.0, 0.8])

    with pytest.raises(ValueError, match="p"):
        expected_complete_log_likelihood(
            times,
            magnitudes,
            rho,
            bg,
            mu=0.5,
            K=0.2,
            alpha=0.5,
            c=0.1,
            p=0.0,
        )