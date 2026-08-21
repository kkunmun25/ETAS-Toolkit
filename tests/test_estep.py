import numpy as np
import pytest

from eq_toolkit.calibrate.estep import EStepResult, compute_estep, make_estep_result


def test_estep_result_valid():
    rho = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.1, 0.3, 0.0],
        ]
    )

    bg = np.array([1.0, 0.8, 0.6])

    result = make_estep_result(rho, bg)

    assert isinstance(result, EStepResult)
    assert result.rho.shape == (3, 3)
    assert result.bg.shape == (3,)

    assert np.allclose(
        result.bg + result.rho.sum(axis=1),
        1.0,
    )


def test_estep_rejects_invalid_probability_sum():
    rho = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
        ]
    )

    bg = np.array([1.0, 0.5])

    with pytest.raises(ValueError, match="invariant"):
        make_estep_result(rho, bg)


def test_estep_rejects_nonzero_diagonal():
    rho = np.array(
        [
            [0.1, 0.0],
            [0.2, 0.0],
        ]
    )

    bg = np.array([0.9, 0.8])

    with pytest.raises(ValueError, match="diagonal"):
        make_estep_result(rho, bg)


def test_estep_rejects_future_triggering():
    rho = np.array(
        [
            [0.0, 0.2],
            [0.0, 0.0],
        ]
    )

    bg = np.array([0.8, 1.0])

    with pytest.raises(ValueError, match="lower triangular"):
        make_estep_result(rho, bg)

def test_compute_estep_shapes_and_invariant():
    times = np.array([0.0, 1.0, 2.0, 4.0])
    magnitudes = np.array([2.0, 2.5, 3.0, 2.2])

    result = compute_estep(
        times,
        magnitudes,
        mu=0.5,
        K=0.2,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    assert result.rho.shape == (4, 4)
    assert result.bg.shape == (4,)

    assert np.allclose(
        result.bg + result.rho.sum(axis=1),
        1.0,
    )


def test_first_event_is_background():
    times = np.array([0.0, 1.0, 2.0])
    magnitudes = np.array([2.0, 2.5, 3.0])

    result = compute_estep(
        times,
        magnitudes,
        mu=0.5,
        K=0.2,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    assert result.bg[0] == 1.0
    assert np.allclose(result.rho[0], 0.0)


def test_future_events_have_zero_trigger_probability():
    times = np.array([0.0, 1.0, 2.0])
    magnitudes = np.array([2.0, 2.5, 3.0])

    result = compute_estep(
        times,
        magnitudes,
        mu=0.5,
        K=0.2,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    assert np.allclose(
        np.triu(result.rho),
        0.0,
    )


def test_triggering_probability_is_nonnegative():
    times = np.array([0.0, 1.0, 2.0])
    magnitudes = np.array([2.0, 2.5, 3.0])

    result = compute_estep(
        times,
        magnitudes,
        mu=0.5,
        K=0.2,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    assert np.all(result.rho >= 0)
    assert np.all(result.bg >= 0)        

def test_estep_rho_is_strictly_lower_triangular():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    magnitudes = np.array([2.0, 2.5, 3.0, 2.2, 2.8])

    result = compute_estep(
        times,
        magnitudes,
        mu=0.5,
        K=0.2,
        alpha=0.5,
        c=0.1,
        p=1.2,
    )

    assert np.allclose(
        np.triu(result.rho),
        0.0,
    )

    assert np.any(result.rho[np.tril_indices(5, k=-1)] > 0)    