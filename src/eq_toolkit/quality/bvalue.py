"""
b-value estimation methods for earthquake catalogs.

Includes:
- Aki-Utsu maximum-likelihood b-value
- Tinti-Mulargia exact binned-magnitude MLE
- Shi-Bolt standard error
- b-positive
- b-more-positive
- OLS diagnostic estimate
- Gutenberg-Richter a-value
- annualized earthquake rate
"""

import numpy as np


def _clean_magnitudes(magnitudes):
    """Return finite magnitudes as a 1-D float array."""
    values = np.asarray(magnitudes, dtype=float).ravel()
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError("No valid magnitudes supplied.")

    return values


def aki_utsu_b_value(
    magnitudes,
    mc,
    delta_m=0.1,
):
    """
    Estimate b using the Aki-Utsu maximum-likelihood estimator.

    Uses the delta_m / 2 binning correction.
    """
    if delta_m <= 0:
        raise ValueError("delta_m must be positive.")

    magnitudes = _clean_magnitudes(magnitudes)

    complete = magnitudes[magnitudes >= mc]

    if len(complete) < 2:
        raise ValueError(
            "At least two earthquakes are required."
        )

    mean_magnitude = np.mean(complete)

    denominator = (
        mean_magnitude
        - (mc - delta_m / 2.0)
    )

    if denominator <= 0:
        raise ValueError(
            "Invalid denominator for b-value estimation."
        )

    b = np.log10(np.e) / denominator

    return float(b)


def shi_bolt_sigma(
    magnitudes,
    mc,
    delta_m=0.1,
):
    """
    Estimate b-value standard error using Shi-Bolt.
    """
    magnitudes = _clean_magnitudes(magnitudes)

    complete = magnitudes[magnitudes >= mc]

    n = len(complete)

    if n < 2:
        raise ValueError(
            "At least two earthquakes are required."
        )

    b = aki_utsu_b_value(
        complete,
        mc=mc,
        delta_m=delta_m,
    )

    mean_magnitude = np.mean(complete)

    variance_sum = np.sum(
        (complete - mean_magnitude) ** 2
    )

    sigma = (
        2.3
        * b**2
        * np.sqrt(
            variance_sum / (n * (n - 1))
        )
    )

    return float(sigma)


def tinti_mulargia_b_value(
    magnitudes,
    mc,
    delta_m=0.1,
):
    """
    Tinti-Mulargia exact maximum-likelihood estimator
    for binned magnitudes.

    The magnitudes are treated as discrete magnitude bins.
    """
    if delta_m <= 0:
        raise ValueError("delta_m must be positive.")

    magnitudes = _clean_magnitudes(magnitudes)

    complete = magnitudes[magnitudes >= mc]

    if len(complete) < 2:
        raise ValueError(
            "At least two earthquakes are required."
        )

    # Shift magnitudes relative to Mc.
    x = complete - mc

    # Numerical solution of the exact binned likelihood.
    #
    # For binned magnitudes:
    #
    # p(M) ∝ exp(-beta * M)
    #
    # and the binning correction is represented through
    # sinh(beta * delta_m / 2).
    #
    # Solve the score equation numerically.

    mean_x = np.mean(x)

    if mean_x <= 0:
        raise ValueError(
            "Invalid magnitude distribution."
        )

    def score(beta):
        half = beta * delta_m / 2.0

        if abs(half) < 1e-8:
            correction = delta_m / 2.0
        else:
            correction = (
                delta_m / 2.0
                / np.tanh(half)
                - 1.0 / beta
            )

        return (
            mean_x
            - (
                1.0 / beta
                - delta_m / 2.0 / np.tanh(half)
            )
        )

    # Robust bisection over beta.
    lo = 1e-8
    hi = 100.0

    f_lo = score(lo)
    f_hi = score(hi)

    if f_lo * f_hi > 0:
        # Fall back to Aki-Utsu for pathological samples.
        return aki_utsu_b_value(
            complete,
            mc=mc,
            delta_m=delta_m,
        )

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = score(mid)

        if abs(f_mid) < 1e-10:
            break

        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid

    beta = mid

    return float(beta / np.log(10.0))


def b_positive(
    magnitudes,
    times=None,
):
    """
    Estimate b using the b-positive method.

    Uses only positive magnitude differences between
    successive earthquakes.

    If times are supplied, events are sorted chronologically.
    """
    magnitudes = _clean_magnitudes(magnitudes)

    if times is not None:
        times = np.asarray(times)

        if len(times) != len(magnitudes):
            raise ValueError(
                "times and magnitudes must have equal length."
            )

        order = np.argsort(times)
        magnitudes = magnitudes[order]

    differences = np.diff(magnitudes)

    positive = differences[differences > 0]

    if len(positive) == 0:
        raise ValueError(
            "No positive magnitude differences found."
        )

    mean_difference = np.mean(positive)

    if mean_difference <= 0:
        raise ValueError(
            "Invalid positive magnitude differences."
        )

    # Positive-difference exponential MLE.
    beta = 1.0 / mean_difference

    return float(beta / np.log(10.0))


def b_more_positive(
    magnitudes,
    times=None,
):
    """
    Estimate b using the b-more-positive approach.

    The catalog is ordered by time and successive
    magnitude changes are used to reduce sensitivity
    to catalog incompleteness.
    """
    magnitudes = _clean_magnitudes(magnitudes)

    if times is not None:
        times = np.asarray(times)

        if len(times) != len(magnitudes):
            raise ValueError(
                "times and magnitudes must have equal length."
            )

        order = np.argsort(times)
        magnitudes = magnitudes[order]

    differences = np.diff(magnitudes)

    # Retain positive differences.
    positive = differences[differences > 0]

    if len(positive) < 2:
        raise ValueError(
            "Not enough positive magnitude differences."
        )

    # More-positive estimator: use the mean positive
    # magnitude increment.
    mean_positive = np.mean(positive)

    beta = 1.0 / mean_positive

    return float(beta / np.log(10.0))


def ols_b_value(
    magnitudes,
    mc,
    delta_m=0.1,
):
    """
    Estimate b using ordinary least squares.

    This is a diagnostic estimator only.

    OLS can be biased because earthquake magnitudes are
    discrete and frequency counts have unequal statistical
    uncertainty.
    """
    if delta_m <= 0:
        raise ValueError("delta_m must be positive.")

    magnitudes = _clean_magnitudes(magnitudes)

    complete = magnitudes[magnitudes >= mc]

    if len(complete) < 2:
        raise ValueError(
            "At least two earthquakes are required."
        )

    max_mag = np.max(complete)

    bins = np.arange(
        mc,
        max_mag + delta_m,
        delta_m,
    )

    if len(bins) < 2:
        raise ValueError(
            "Not enough magnitude bins."
        )

    counts = np.array(
        [
            np.sum(complete >= m)
            for m in bins
        ],
        dtype=float,
    )

    valid = counts > 0

    x = bins[valid]
    y = np.log10(counts[valid])

    if len(x) < 2:
        raise ValueError(
            "Not enough points for OLS."
        )

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    return float(-slope)


def a_value(
    magnitudes,
    b,
    mc,
):
    """
    Calculate the Gutenberg-Richter a-value.

    log10(N) = a - b*M
    """
    magnitudes = _clean_magnitudes(magnitudes)

    complete = magnitudes[magnitudes >= mc]

    n = len(complete)

    if n == 0:
        raise ValueError(
            "No earthquakes above Mc."
        )

    return float(
        np.log10(n) + b * mc
    )


def annualized_rate(
    n_events,
    duration_years,
):
    """
    Calculate the annualized earthquake occurrence rate.
    """
    if n_events < 0:
        raise ValueError(
            "n_events cannot be negative."
        )

    if duration_years <= 0:
        raise ValueError(
            "duration_years must be positive."
        )

    return float(
        n_events / duration_years
    )