"""
Temporal ETAS log-likelihood.

Implements Ogata's incomplete log-likelihood
for a temporal ETAS model.
"""

import numpy as np

from .intensity import temporal_intensity
from .kernels import omori_integral


def temporal_log_likelihood(
    times,
    magnitudes,
    mu,
    K,
    alpha,
    M0,
    c,
    p,
    t_start=None,
    t_end=None,
):
    """
    Calculate the incomplete temporal ETAS log-likelihood.

    log L =
        sum(log(lambda(t_i)))
        - integral(lambda(t) dt)

    over [t_start, t_end].

    Parameters
    ----------
    mu : Background event rate.

    K : Productivity parameter.

    alpha : Magnitude productivity parameter.

    M0 : Reference magnitude.

    c : Omori time offset.

    p : Omori exponent. Must be > 1.

    t_start : Beginning of observation window.

    t_end : End of observation window.

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

    if len(times) == 0:
        raise ValueError("times cannot be empty.")

    if mu <= 0:
        raise ValueError("mu must be positive.")

    if K < 0:
        raise ValueError("K must be non-negative.")

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    if c <= 0:
        raise ValueError("c must be positive.")

    if p <= 1:
        raise ValueError("p must be greater than 1.")

    # Observation window
    if t_start is None:
        t_start = float(np.min(times))

    if t_end is None:
        t_end = float(np.max(times))

    if t_end <= t_start:
        raise ValueError(
            "t_end must be greater than t_start."
        )

    # Only events inside the observation window
    # contribute to the event term.
    mask = (times >= t_start) & (times <= t_end)

    event_times = times[mask]
    event_magnitudes = magnitudes[mask]

    if len(event_times) == 0:raise ValueError( "No events fall inside the observation window." )

    # 1. EVENT TERM
    # sum log(lambda(t_i))

    intensities = temporal_intensity(
        event_times,
        event_magnitudes,
        mu=mu,
        K=K,
        alpha=alpha,
        M0=M0,
        c=c,
        p=p,
    )

    if np.any(intensities <= 0):
        return -np.inf

    event_term = np.sum(np.log(intensities))


    # 2. COMPENSATOR
    # integral lambda(t) dt
    # Background: mu * (t_end - t_start)
    # Triggered contribution: K * productivity * integral Omori
    
    background_integral = mu * (t_end - t_start)

    triggered_integral = 0.0

    for ti, magnitude in zip(event_times, event_magnitudes):

        # An event can only contribute after it occurs.
        if ti >= t_end:
            continue

        lower = max(0.0, t_start - ti)
        upper = t_end - ti

        if upper <= lower:
            continue

        productivity = 10.0 ** (
            alpha * (magnitude - M0)
        )

        triggered_integral += (
            K
            * productivity
            * omori_integral(
                lower,
                upper,
                c=c,
                p=p,
            )
        )

    compensator = (background_integral+ triggered_integral)

    # 3. LOG-LIKELIHOOD

    return float(event_term - compensator)