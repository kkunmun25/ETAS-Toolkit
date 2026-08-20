import numpy as np
import pytest

from eq_toolkit.model.intensity import temporal_intensity


def test_first_event_has_background_intensity():
    times = np.array([0.0, 1.0, 2.0])
    magnitudes = np.array([3.0, 3.0, 3.0])

    result = temporal_intensity(
        times,
        magnitudes,
        mu=0.5,
        K=0.1,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
    )

    assert np.isclose(result[0], 0.5)


def test_aftershock_increases_intensity():
    times = np.array([0.0, 1.0])
    magnitudes = np.array([4.0, 3.0])

    result = temporal_intensity(
        times,
        magnitudes,
        mu=0.5,
        K=0.1,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
    )

    assert result[1] > result[0]


def test_coincident_timestamps_are_causal():
    times = np.array([0.0, 0.0, 1.0])
    magnitudes = np.array([4.0, 5.0, 3.0])

    result = temporal_intensity(
        times,
        magnitudes,
        mu=0.5,
        K=0.1,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
    )

    # Events occurring at exactly the same time
    # must NOT trigger each other.
    assert np.isclose(result[0], 0.5)
    assert np.isclose(result[1], 0.5)

    # The event at t=1 can see both previous events.
    assert result[2] > 0.5


def test_empty_input():
    result = temporal_intensity(
        np.array([]),
        np.array([]),
        mu=0.5,
        K=0.1,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
    )

    assert len(result) == 0


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        temporal_intensity(
            np.array([0.0, 1.0]),
            np.array([3.0]),
            mu=0.5,
            K=0.1,
            alpha=1.0,
            M0=3.0,
            c=0.1,
            p=1.2,
        )


def test_nonnegative_intensity():
    times = np.array([0.0, 1.0, 2.0])
    magnitudes = np.array([3.0, 4.0, 3.5])

    result = temporal_intensity(
        times,
        magnitudes,
        mu=0.5,
        K=0.1,
        alpha=1.0,
        M0=3.0,
        c=0.1,
        p=1.2,
    )

    assert np.all(result >= 0)