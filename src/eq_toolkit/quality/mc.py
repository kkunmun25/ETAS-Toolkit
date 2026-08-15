"""
Magnitude of Completeness (Mc) estimation methods.

Methods
-------
MAXC
    Maximum Curvature.

GFT
    Goodness-of-Fit Test.

MBS
    Magnitude of Completeness by b-value Stability.

References
----------
Wiemer, S. & Wyss, M. (2000).
Minimum magnitude of completeness in earthquake catalogs:
examples from Alaska, the Western United States, and Japan.
BSSA, 90(4), 859-869.

Cao, A. & Gao, S. S. (2002).
Temporal variations of seismic b-values beneath northeastern
Japan island arc.
Geophysical Research Letters, 29.

Woessner, J. & Wiemer, S. (2005).
Assessing the quality of earthquake catalogues:
Estimating the magnitude of completeness and its uncertainty.
BSSA, 95(2), 684-698.

Shi, Y. & Bolt, B. A. (1982).
The standard error of the magnitude-frequency b value.
Bulletin of the Seismological Society of America.
"""

import numpy as np


# =====================================================================
# MAXC
# =====================================================================

def maxc(magnitudes, bin_width=0.1):
    """
    Estimate Mc using Maximum Curvature.

    An empirical +0.2 correction is applied following
    Wiemer & Wyss (2000).
    """

    magnitudes = np.asarray(magnitudes, dtype=float)

    magnitudes = magnitudes[np.isfinite(magnitudes)]

    if len(magnitudes) == 0:
        raise ValueError("No valid magnitudes supplied.")

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    min_mag = (
        np.floor(magnitudes.min() / bin_width)
        * bin_width
    )

    max_mag = (
        np.ceil(magnitudes.max() / bin_width)
        * bin_width
    )

    bins = np.arange(
        min_mag,
        max_mag + bin_width,
        bin_width,
    )

    counts, edges = np.histogram(
        magnitudes,
        bins=bins,
    )

    if len(counts) == 0:
        raise ValueError(
            "Unable to construct magnitude bins."
        )

    max_index = np.argmax(counts)

    mc_max_curvature = edges[max_index]

    mc = mc_max_curvature + 0.2

    return float(mc)


# =====================================================================
# B-VALUE
# =====================================================================

def b_value(magnitudes, mc, bin_width=0.1):
    """
    Calculate the Gutenberg-Richter maximum-likelihood b-value.

    b = log10(e) /
        [mean(M) - (Mc - delta_M / 2)]
    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    if len(magnitudes) == 0:
        raise ValueError(
            "No valid magnitudes supplied."
        )

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    complete = magnitudes[
        magnitudes >= mc
    ]

    if len(complete) < 2:
        raise ValueError(
            "Not enough earthquakes above Mc "
            "to calculate b-value."
        )

    mean_magnitude = np.mean(complete)

    denominator = (
        mean_magnitude
        - (mc - bin_width / 2.0)
    )

    if denominator <= 0:
        raise ValueError(
            "Invalid denominator while calculating "
            "b-value."
        )

    b = np.log10(np.e) / denominator

    return float(b)


# =====================================================================
# B-VALUE UNCERTAINTY
# =====================================================================

def b_value_sigma(
    magnitudes,
    mc,
    bin_width=0.1,
):
    """
    Calculate the standard uncertainty of the b-value.

    Shi & Bolt (1982):

        sigma_b =
            2.3 * b^2 *
            sqrt(
                sum((Mi - Mmean)^2)
                /
                [N * (N - 1)]
            )

    Parameters
    ----------
    magnitudes : array-like
        Earthquake magnitudes.

    mc : float
        Candidate magnitude of completeness.

    bin_width : float, default=0.1
        Magnitude bin width.

    Returns
    -------
    float
        Standard uncertainty of b.
    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    complete = magnitudes[
        magnitudes >= mc
    ]

    n = len(complete)

    if n < 2:
        raise ValueError(
            "At least two earthquakes are required "
            "to calculate b-value uncertainty."
        )

    b = b_value(
        complete,
        mc=mc,
        bin_width=bin_width,
    )

    mean_magnitude = np.mean(complete)

    squared_deviations = np.sum(
        (complete - mean_magnitude) ** 2
    )

    sigma_b = (
        2.3
        * b**2
        * np.sqrt(
            squared_deviations
            /
            (n * (n - 1))
        )
    )

    return float(sigma_b)


# =====================================================================
# GFT INTERNAL FUNCTION
# =====================================================================

def _gft_score(
    magnitudes,
    mc,
    bin_width=0.1,
):
    """
    Calculate Gutenberg-Richter goodness-of-fit percentage.
    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    complete = magnitudes[
        magnitudes >= mc
    ]

    if len(complete) < 2:
        return np.nan, np.nan

    b = b_value(
        complete,
        mc=mc,
        bin_width=bin_width,
    )

    max_mag = np.max(complete)

    thresholds = np.arange(
        mc,
        max_mag + bin_width / 2.0,
        bin_width,
    )

    thresholds = np.round(
        thresholds,
        10,
    )

    if len(thresholds) == 0:
        return np.nan, np.nan

    observed = np.array([
        np.sum(complete >= m)
        for m in thresholds
    ], dtype=float)

    if observed[0] <= 0:
        return np.nan, np.nan

    predicted = (
        observed[0]
        * 10.0 ** (
            -b * (thresholds - mc)
        )
    )

    misfit = np.sum(
        np.abs(
            observed - predicted
        )
    )

    total_observed = np.sum(
        observed
    )

    if total_observed <= 0:
        return np.nan, np.nan

    goodness = (
        100.0
        * (
            1.0
            - misfit / total_observed
        )
    )

    goodness = max(
        0.0,
        min(100.0, goodness),
    )

    return (
        float(goodness),
        float(b),
    )


# =====================================================================
# GFT
# =====================================================================

def gft(
    magnitudes,
    bin_width=0.1,
    min_fit=0.95,
    fallback_fit=0.90,
    min_events=50,
):
    """
    Estimate Mc using the Goodness-of-Fit Test.

    95% goodness-of-fit is attempted first.
    90% is used as fallback.
    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    if len(magnitudes) == 0:
        raise ValueError(
            "No valid magnitudes supplied."
        )

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    if not (
        0 < fallback_fit <= min_fit <= 1
    ):
        raise ValueError(
            "Thresholds must satisfy "
            "0 < fallback_fit <= min_fit <= 1."
        )

    if min_events < 2:
        raise ValueError(
            "min_events must be at least 2."
        )

    min_mag = (
        np.floor(
            magnitudes.min()
            / bin_width
        )
        * bin_width
    )

    max_mag = (
        np.floor(
            magnitudes.max()
            / bin_width
        )
        * bin_width
    )

    candidates = np.arange(
        min_mag,
        max_mag + bin_width / 2.0,
        bin_width,
    )

    candidates = np.round(
        candidates,
        10,
    )

    # ---------------------------------------------------------------
    # 95% criterion
    # ---------------------------------------------------------------

    for candidate in candidates:

        n_events = np.sum(
            magnitudes >= candidate
        )

        if n_events < min_events:
            continue

        goodness, _ = _gft_score(
            magnitudes,
            mc=candidate,
            bin_width=bin_width,
        )

        if np.isfinite(goodness):

            if goodness >= min_fit * 100.0:
                return float(candidate)

    # ---------------------------------------------------------------
    # 90% fallback
    # ---------------------------------------------------------------

    for candidate in candidates:

        n_events = np.sum(
            magnitudes >= candidate
        )

        if n_events < min_events:
            continue

        goodness, _ = _gft_score(
            magnitudes,
            mc=candidate,
            bin_width=bin_width,
        )

        if np.isfinite(goodness):

            if goodness >= fallback_fit * 100.0:
                return float(candidate)

    raise ValueError(
        "No magnitude of completeness satisfied "
        "the 95% or 90% goodness-of-fit criteria."
    )


# =====================================================================
# MBS
# =====================================================================

def mbs(
    magnitudes,
    bin_width=0.1,
    window=0.5,
    min_events=50,
):
    """
    Estimate Mc using the Magnitude of Completeness
    by b-value Stability (MBS).

    This follows the Woessner & Wiemer (2005)
    refinement of the Cao & Gao (2002) method.

    The method:

    1. Try different cutoff magnitudes Mco.
    2. Calculate b(Mco).
    3. Calculate sigma_b using Shi & Bolt (1982).
    4. Calculate the average b-value in a window
       beginning at Mco.
    5. Select the first Mco for which:

           |b(Mco) - b_average| <= sigma_b

    Parameters
    ----------
    magnitudes : array-like
        Earthquake magnitudes.

    bin_width : float, default=0.1
        Magnitude bin size.

    window : float, default=0.5
        Magnitude window used to calculate
        the average b-value.

    min_events : int, default=50
        Minimum number of earthquakes required
        above a candidate cutoff.

    Returns
    -------
    float
        Estimated magnitude of completeness.
    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        np.isfinite(magnitudes)
    ]

    if len(magnitudes) == 0:
        raise ValueError(
            "No valid magnitudes supplied."
        )

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    if min_events < 2:
        raise ValueError(
            "min_events must be at least 2."
        )

    # ---------------------------------------------------------------
    # Candidate cutoff magnitudes
    # ---------------------------------------------------------------

    min_mag = (
        np.floor(
            magnitudes.min()
            / bin_width
        )
        * bin_width
    )

    max_mag = (
        np.floor(
            magnitudes.max()
            / bin_width
        )
        * bin_width
    )

    candidates = np.arange(
        min_mag,
        max_mag + bin_width / 2.0,
        bin_width,
    )

    candidates = np.round(
        candidates,
        10,
    )

    # ---------------------------------------------------------------
    # Calculate b-values for all usable candidates
    # ---------------------------------------------------------------

    b_values = {}
    b_sigmas = {}

    for candidate in candidates:

        n_events = np.sum(
            magnitudes >= candidate
        )

        if n_events < min_events:
            continue

        try:

            b = b_value(
                magnitudes,
                mc=candidate,
                bin_width=bin_width,
            )

            sigma = b_value_sigma(
                magnitudes,
                mc=candidate,
                bin_width=bin_width,
            )

        except ValueError:
            continue

        if (
            np.isfinite(b)
            and np.isfinite(sigma)
        ):

            b_values[
                float(candidate)
            ] = b

            b_sigmas[
                float(candidate)
            ] = sigma

    if len(b_values) < 2:
        raise ValueError(
            "Not enough usable magnitude bins "
            "for MBS estimation."
        )

    candidate_values = sorted(
        b_values.keys()
    )

    # ---------------------------------------------------------------
    # Search for the stable b-value region
    # ---------------------------------------------------------------

    for candidate in candidate_values:

        upper_limit = (
            candidate + window
        )

        window_b_values = [
            b_values[m]
            for m in candidate_values
            if (
                m >= candidate
                and m <= upper_limit + 1e-10
            )
        ]

        # We need at least two b-values
        # to define a local average.
        if len(window_b_values) < 2:
            continue

        b_average = np.mean(
            window_b_values
        )

        difference = abs(
            b_values[candidate]
            - b_average
        )

        sigma_b = b_sigmas[
            candidate
        ]

        # Woessner & Wiemer stability criterion
        if difference <= sigma_b:

            return float(candidate)

    raise ValueError(
        "No stable b-value region found "
        "for the supplied catalog."
    )