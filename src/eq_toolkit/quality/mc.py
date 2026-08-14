"""
Magnitude of Completeness (Mc) estimation methods.

Implemented methods:
    - MAXC: Maximum Curvature
    - GFT: Goodness-of-Fit Test 
"""

import numpy as np


# ---------------------------------------------------------------------
# MAXC
# ---------------------------------------------------------------------

def maxc(magnitudes, bin_width=0.1):
    """
    Estimate magnitude of completeness using Maximum Curvature.

    The maximum-curvature magnitude is the magnitude bin containing
    the largest number of earthquakes. An empirical +0.2 correction
    is then applied following Wiemer & Wyss (2000).

   
    """

    magnitudes = np.asarray(magnitudes, dtype=float)

    # Remove NaN and infinite values
    magnitudes = magnitudes[np.isfinite(magnitudes)]

    if len(magnitudes) == 0:
        raise ValueError("No valid magnitudes supplied.")

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    # Construct magnitude bins
    min_mag = np.floor(magnitudes.min() / bin_width) * bin_width
    max_mag = np.ceil(magnitudes.max() / bin_width) * bin_width

    bins = np.arange(
        min_mag,
        max_mag + bin_width,
        bin_width
    )

    # Histogram
    counts, edges = np.histogram(magnitudes, bins=bins)

    if len(counts) == 0:
        raise ValueError("Unable to construct magnitude bins.")

    # Find maximum-frequency bin
    max_index = np.argmax(counts)

    # Magnitude at maximum curvature
    mc_max_curvature = edges[max_index]

    # Empirical correction
    mc = mc_max_curvature + 0.2

    return float(mc)


# ---------------------------------------------------------------------
# B-VALUE
# ---------------------------------------------------------------------

def b_value(magnitudes, mc, bin_width=0.1):
    """
    Calculate the Gutenberg-Richter b-value using maximum likelihood.

    Formula:

        b = log10(e) /
            (mean(M) - (Mc - delta_M / 2))

    where:
        M       = magnitudes >= Mc
        Mc      = candidate magnitude of completeness
        delta_M = magnitude bin width

    """

    magnitudes = np.asarray(magnitudes, dtype=float)

    # Remove invalid values
    magnitudes = magnitudes[np.isfinite(magnitudes)]

    if len(magnitudes) == 0:
        raise ValueError("No valid magnitudes supplied.")

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    # Select complete part of catalog
    complete = magnitudes[magnitudes >= mc]

    if len(complete) < 2:
        raise ValueError(
            "Not enough earthquakes above Mc to calculate b-value."
        )

    mean_magnitude = np.mean(complete)

    denominator = mean_magnitude - (mc - bin_width / 2.0)

    if denominator <= 0:
        raise ValueError(
            "Invalid denominator while calculating b-value."
        )

    # log10(e)
    b = np.log10(np.e) / denominator

    return float(b)


# ---------------------------------------------------------------------
# GFT HELPER
# ---------------------------------------------------------------------

def _gft_score(magnitudes, mc, bin_width=0.1):
    """
    Calculate Gutenberg-Richter goodness-of-fit percentage.


    """

    magnitudes = np.asarray(magnitudes, dtype=float)

    # Earthquakes above candidate Mc
    complete = magnitudes[magnitudes >= mc]

    if len(complete) < 2:
        return np.nan, np.nan

    # Calculate b-value
    b = b_value(
        complete,
        mc=mc,
        bin_width=bin_width
    )

    # Magnitude thresholds used for cumulative distribution
    max_mag = np.max(complete)

    thresholds = np.arange(
        mc,
        max_mag + bin_width / 2.0,
        bin_width
    )

    # Avoid floating-point issues such as 2.5000000000000004
    thresholds = np.round(thresholds, 10)

    if len(thresholds) == 0:
        return np.nan, np.nan

    # Observed cumulative number of earthquakes
    observed = np.array([
        np.sum(complete >= m)
        for m in thresholds
    ], dtype=float)

    if observed[0] <= 0:
        return np.nan, np.nan

    # Gutenberg-Richter predicted cumulative numbers
    #
    # N(M) = N(Mc) * 10^[-b(M-Mc)]
    predicted = (
        observed[0]
        * 10.0 ** (-b * (thresholds - mc))
    )

    # Wiemer & Wyss style goodness-of-fit measure
    #
    # GFT = 100 * [1 -
    #       sum(|observed - predicted|)
    #       / sum(observed)]
    #
    misfit = np.sum(np.abs(observed - predicted))

    total_observed = np.sum(observed)

    if total_observed <= 0:
        return np.nan, np.nan

    goodness = 100.0 * (
        1.0 - misfit / total_observed
    )

    # Keep result within sensible range
    goodness = max(0.0, min(100.0, goodness))

    return float(goodness), float(b)


# ---------------------------------------------------------------------
# GFT
# ---------------------------------------------------------------------

def gft(
    magnitudes,
    bin_width=0.1,
    min_fit=0.95,
    fallback_fit=0.90,
    min_events=50,
):
    """
    Estimate magnitude of completeness using the
    Goodness-of-Fit Test (GFT).

    The method tests possible Mc values and selects the
    smallest magnitude for which the Gutenberg-Richter
    distribution reaches the required goodness-of-fit.

    First a 95% fit is searched.

    If no candidate reaches 95%, a 90% fallback is used.

   
    """

    magnitudes = np.asarray(magnitudes, dtype=float)

    # Remove invalid values
    magnitudes = magnitudes[np.isfinite(magnitudes)]

    if len(magnitudes) == 0:
        raise ValueError("No valid magnitudes supplied.")

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    if not (0 < fallback_fit <= min_fit <= 1):
        raise ValueError(
            "Thresholds must satisfy "
            "0 < fallback_fit <= min_fit <= 1."
        )

    if min_events < 2:
        raise ValueError("min_events must be at least 2.")

    # Candidate Mc values
    min_mag = (
        np.floor(magnitudes.min() / bin_width)
        * bin_width
    )

    max_mag = (
        np.floor(magnitudes.max() / bin_width)
        * bin_width
    )

    candidates = np.arange(
        min_mag,
        max_mag + bin_width / 2.0,
        bin_width
    )

    # Fix floating-point values such as:
    # 2.5000000000000004
    candidates = np.round(candidates, 10)

    # -------------------------------------------------------------
    # First attempt: 95%
    # -------------------------------------------------------------

    for candidate in candidates:

        n_events = np.sum(magnitudes >= candidate)

        if n_events < min_events:
            continue

        goodness, _ = _gft_score(
            magnitudes,
            mc=candidate,
            bin_width=bin_width
        )

        if np.isfinite(goodness):

            if goodness >= min_fit * 100.0:
                return float(candidate)

    # -------------------------------------------------------------
    # Fallback: 90%
    # -------------------------------------------------------------

    for candidate in candidates:

        n_events = np.sum(magnitudes >= candidate)

        if n_events < min_events:
            continue

        goodness, _ = _gft_score(
            magnitudes,
            mc=candidate,
            bin_width=bin_width
        )

        if np.isfinite(goodness):

            if goodness >= fallback_fit * 100.0:
                return float(candidate)

    # -------------------------------------------------------------
    # Nothing satisfied either threshold
    # -------------------------------------------------------------

    raise ValueError(
        "No magnitude of completeness satisfied "
        "the 95% or 90% goodness-of-fit criteria."
    )