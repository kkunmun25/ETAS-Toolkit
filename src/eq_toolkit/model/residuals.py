"""
Residual diagnostics for the temporal ETAS model.

Implements Ogata transformed-time residuals and
a Kolmogorov-Smirnov diagnostic.
"""

import numpy as np
from scipy.stats import kstest

from .kernels import omori_integral


def transformed_time_residuals(
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
    Calculate Ogata transformed-time residuals.

    For consecutive events:

        tau_i = integral_{t_{i-1}}^{t_i} lambda(t) dt

    Under a correctly specified point-process model,
    the transformed inter-event times should follow
    an exponential distribution with mean 1.

    Parameters
    ----------
    mu : background rate.

    K : Productivity parameter.

    alpha : Magnitude productivity parameter.

    M0 : Reference magnitude.

    c : Omori time offset.

    p : Omori exponent.

    """

    times = np.asarray(times, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)

    if times.ndim != 1:
        raise ValueError("times must be one-dimensional.")

    if magnitudes.ndim != 1:
        raise ValueError(
            "magnitudes must be one-dimensional."
        )

    if len(times) != len(magnitudes):
        raise ValueError(
            "times and magnitudes must have the same length."
        )

    if len(times) < 2:
        raise ValueError(
            "At least two events are required."
        )

    if mu <= 0:
        raise ValueError("mu must be positive.")

    if K < 0:
        raise ValueError("K must be non-negative.")

    if alpha < 0:
        raise ValueError(
            "alpha must be non-negative."
        )

    if c <= 0:
        raise ValueError("c must be positive.")

    if p <= 1:
        raise ValueError(
            "p must be greater than 1."
        )

    if np.any(np.diff(times) < 0):
        raise ValueError(
            "times must be sorted in ascending order."
        )

    residuals = np.zeros(len(times) - 1)

    # Calculate the transformed time between consecutive events
    

    for i in range(1, len(times)):

        t_previous = times[i - 1]
        t_current = times[i]

        dt = t_current - t_previous

        # Background contribution
        background = mu * dt

        # Triggered contribution
        triggered = 0.0

        # Every earthquake before t_current can contribute.
        for j in range(i):

            ti = times[j]
            magnitude = magnitudes[j]

            lower = t_previous - ti
            upper = t_current - ti

            # The integration starts only after the
            # triggering earthquake.
            lower = max(0.0, lower)

            if upper <= lower:
                continue

            productivity = 10.0 ** (
                alpha * (magnitude - M0)
            )

            triggered += (
                K
                * productivity
                * omori_integral(
                    lower,
                    upper,
                    c=c,
                    p=p,
                )
            )

        residuals[i - 1] = (
            background + triggered
        )

    return residuals


def ks_test_residuals(residuals):
    """
    Perform a Kolmogorov-Smirnov test against
    an Exponential(1) distribution.

    """

    residuals = np.asarray(
        residuals,
        dtype=float,
    )

    if residuals.ndim != 1:
        raise ValueError(
            "residuals must be one-dimensional."
        )

    if len(residuals) == 0:
        raise ValueError(
            "residuals cannot be empty."
        )

    if np.any(residuals < 0):
        raise ValueError(
            "residuals must be non-negative."
        )

    statistic, p_value = kstest(
        residuals,
        "expon",
    )

    return float(statistic), float(p_value)