"""
Temporal conditional intensity for the ETAS model.
"""

import numpy as np

from .kernels import omori_kernel


def temporal_intensity(
    times,
    magnitudes,
    mu,
    K,
    alpha,
    M0,
    c,
    p,
):
    """
    Calculate the temporal ETAS conditional intensity
    at every event time.

    Parameters
    ----------

    mu : Background event rate.

    K :  Productivity parameter.

    alpha : Magnitude productivity parameter.

    M0 : Reference magnitude.

    c : Omori time offset.

    p : Omori decay exponent.

    """

    times = np.asarray(times, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)

    if times.ndim != 1:
        raise ValueError("times must be one-dimensional.")

    if magnitudes.ndim != 1:
        raise ValueError("magnitudes must be one-dimensional.")

    if len(times) != len(magnitudes):
        raise ValueError(
            "times and magnitudes must have the same length."
        )

    if mu < 0:
        raise ValueError("mu must be non-negative.")

    if K < 0:
        raise ValueError("K must be non-negative.")

    if len(times) == 0:
        return np.array([], dtype=float)

    intensity = np.full(len(times), float(mu))

    for i in range(len(times)):

        # Only earthquakes that occurred BEFORE
        # the current earthquake can contribute.
        mask = times < times[i]

        if not np.any(mask):
            continue

        dt = times[i] - times[mask]

        productivity = 10.0 ** (
            alpha * (magnitudes[mask] - M0)
        )

        intensity[i] += K * np.sum(
            productivity * omori_kernel(dt, c, p)
        )

    return intensity