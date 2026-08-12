"""
Frequency-magnitude distribution (FMD) utilities.

This module provides a simple exploratory FMD for an earthquake Catalog.

Scientific background:
    Gutenberg-Richter relation:
        log10(N) = a - b*M

    Aki-Utsu maximum-likelihood b-value:
        b = log10(e) / (mean(M) - Mc + delta_M/2)

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_dataframe(catalog) -> pd.DataFrame:
   
    if isinstance(catalog, pd.DataFrame):
        return catalog.copy()

    if hasattr(catalog, "data"):
        data = catalog.data
        if isinstance(data, pd.DataFrame):
            return data.copy()

    if hasattr(catalog, "df"):
        data = catalog.df
        if isinstance(data, pd.DataFrame):
            return data.copy()

    raise TypeError(
        "catalog must be a pandas DataFrame or a Catalog containing "
        "a pandas DataFrame in .data or .df"
    )


def _get_magnitudes(catalog) -> np.ndarray:
  
    df = _get_dataframe(catalog)

    if "magnitude" not in df.columns:
        raise ValueError(
            "Catalog must contain a 'magnitude' column."
        )

    magnitudes = pd.to_numeric(
        df["magnitude"], errors="coerce"
    ).dropna()

    magnitudes = magnitudes[np.isfinite(magnitudes)]

    if len(magnitudes) == 0:
        raise ValueError("Catalog contains no valid earthquake magnitudes.")

    return magnitudes.to_numpy(dtype=float)


def _estimate_mc(
    magnitudes: np.ndarray,
    delta_m: float = 0.1,
) -> float:
    """
    Estimate magnitude of completeness using maximum curvature.

    """
    if magnitudes.size == 0:
        raise ValueError("Cannot estimate Mc from an empty catalog.")

    min_mag = np.floor(magnitudes.min() / delta_m) * delta_m
    max_mag = np.ceil(magnitudes.max() / delta_m) * delta_m

    if max_mag <= min_mag:
        return float(min_mag)

    bins = np.arange(
        min_mag,
        max_mag + delta_m,
        delta_m,
    )

    counts, edges = np.histogram(
        magnitudes,
        bins=bins,
    )

    if counts.size == 0:
        return float(min_mag)

    max_index = int(np.argmax(counts))

    mc = edges[max_index]

    return float(np.round(mc, 6))


def _estimate_b_value(
    magnitudes: np.ndarray,
    mc: float,
    delta_m: float = 0.1,
) -> float:
    """
    Estimate the Gutenberg-Richter b-value using Aki-Utsu MLE.

    Formula:
        b = log10(e) / (mean(M) - Mc + delta_M/2)

    Only earthquakes with M >= Mc are used.
    """
    complete = magnitudes[magnitudes >= mc]

    if len(complete) < 2:
        raise ValueError(
            "At least two earthquakes with magnitude >= Mc are required "
            "to estimate the b-value."
        )

    denominator = np.mean(complete) - mc + delta_m / 2.0

    if denominator <= 0:
        raise ValueError(
            "Cannot estimate b-value because the MLE denominator "
            "is not positive."
        )

    b_value = np.log10(np.e) / denominator

    return float(b_value)


def _estimate_a_value(
    magnitudes: np.ndarray,
    mc: float,
    b_value: float,
) -> float:
    """
    Estimate the Gutenberg-Richter a-value.

    Using:
        log10(N) = a - b*M

    at M = Mc:
        a = log10(N) + b*Mc

    where N is the number of events with M >= Mc.
    """
    complete_count = int(np.sum(magnitudes >= mc))

    if complete_count <= 0:
        raise ValueError(
            "No earthquakes are available at or above Mc."
        )

    a_value = np.log10(complete_count) + b_value * mc

    return float(a_value)


def _frequency_magnitude_data(
    magnitudes: np.ndarray,
    delta_m: float = 0.1,
):
    """
    Calculate incremental and cumulative FMD values.
    """
    min_mag = np.floor(magnitudes.min() / delta_m) * delta_m
    max_mag = np.ceil(magnitudes.max() / delta_m) * delta_m

    magnitude_values = np.arange(
        min_mag,
        max_mag + delta_m / 2,
        delta_m,
    )

    incremental = np.zeros_like(magnitude_values, dtype=int)

    for i, magnitude in enumerate(magnitude_values):
        lower = magnitude - delta_m / 2.0
        upper = magnitude + delta_m / 2.0

        if i == 0:
            mask = magnitudes < upper
        else:
            mask = (magnitudes >= lower) & (magnitudes < upper)

        incremental[i] = int(np.sum(mask))

    cumulative = np.array(
        [
            np.sum(magnitudes >= magnitude)
            for magnitude in magnitude_values
        ],
        dtype=int,
    )

    return magnitude_values, incremental, cumulative


def plot_fmd(
    catalog,
    mc: float | None = None,
    delta_m: float = 0.1,
    ax=None,
):
    """
    Plot the frequency-magnitude distribution for an earthquake Catalog.

    Parameters
    ----------
    catalog
        Project Catalog object or pandas DataFrame containing a
        'magnitude' column.

    mc : float, optional
        Magnitude of completeness. If None, estimate Mc using
        maximum curvature.

    delta_m : float, default=0.1
        Magnitude bin width.

    ax : matplotlib.axes.Axes, optional
        Existing axes on which to draw the FMD.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the FMD.

    Notes
    -----
    The Gutenberg-Richter relation is:

        log10(N) = a - b*M

    The b-value is estimated using the Aki-Utsu maximum-likelihood
    estimator with the half-bin correction.

    The formal completeness methods are implemented separately in
    Phase 4 under quality/mc.py.
    """
    magnitudes = _get_magnitudes(catalog)

    if delta_m <= 0:
        raise ValueError("delta_m must be greater than zero.")

    if mc is None:
        mc = _estimate_mc(
            magnitudes,
            delta_m=delta_m,
        )

    mc = float(mc)

    magnitude_values, incremental, cumulative = (
        _frequency_magnitude_data(
            magnitudes,
            delta_m=delta_m,
        )
    )

    b_value = _estimate_b_value(
        magnitudes,
        mc,
        delta_m=delta_m,
    )

    a_value = _estimate_a_value(
        magnitudes,
        mc,
        b_value,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Incremental FMD
    incremental_positive = incremental > 0

    ax.plot(
        magnitude_values[incremental_positive],
        incremental[incremental_positive],
        "o",
        label="Incremental",
        alpha=0.75,
    )

    # Cumulative FMD
    cumulative_positive = cumulative > 0

    ax.plot(
        magnitude_values[cumulative_positive],
        cumulative[cumulative_positive],
        "s-",
        label="Cumulative",
        alpha=0.8,
    )

    # Gutenberg-Richter fitted line
    fit_min = mc
    fit_max = magnitude_values.max()

    fit_magnitudes = np.linspace(
        fit_min,
        fit_max,
        100,
    )

    fit_counts = 10 ** (
        a_value - b_value * fit_magnitudes
    )

    positive_fit = fit_counts > 0

    ax.plot(
        fit_magnitudes[positive_fit],
        fit_counts[positive_fit],
        "--",
        linewidth=2,
        label="Gutenberg-Richter fit",
    )

    # Mc marker
    ax.axvline(
        mc,
        linestyle=":",
        linewidth=2,
        label=fr"$M_c={mc:.1f}$",
    )

    # Scientific labels
    ax.set_yscale("log")

    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Number of earthquakes")
    ax.set_title("Frequency-Magnitude Distribution")

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend()

    # b/a annotation
    annotation = (
        rf"$M_c={mc:.2f}$"
        "\n"
        rf"$b={b_value:.3f}$"
        "\n"
        rf"$a={a_value:.3f}$"
        "\n"
        rf"$N(M\geq M_c)={np.sum(magnitudes >= mc)}$"
    )

    ax.text(
        0.97,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
        ),
    )

    return ax


__all__ = [
    "plot_fmd",
]