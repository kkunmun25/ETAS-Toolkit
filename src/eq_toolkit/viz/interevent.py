"""
Inter-event time visualization.

Calculates and plots the time between consecutive earthquakes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_dataframe(catalog) -> pd.DataFrame:
    """Extract the DataFrame from the project Catalog."""
    if isinstance(catalog, pd.DataFrame):
        return catalog.copy()

    if hasattr(catalog, "data"):
        if isinstance(catalog.data, pd.DataFrame):
            return catalog.data.copy()

    raise TypeError(
        "catalog must be a pandas DataFrame or a Catalog "
        "containing a pandas DataFrame in .data"
    )


def _get_interevent_times(catalog) -> pd.Series:
    """
    Calculate time differences between consecutive earthquakes.

    Returns
    -------
    pandas.Series
        Inter-event times in hours.
    """
    df = _get_dataframe(catalog)

    if "time" not in df.columns:
        raise ValueError(
            "Catalog must contain a 'time' column."
        )

    times = pd.to_datetime(
        df["time"],
        errors="coerce",
        utc=True,
    ).dropna()

    times = times.sort_values()

    if len(times) < 2:
        raise ValueError(
            "At least two earthquake events are required "
            "to calculate inter-event times."
        )

    differences = times.diff().dropna()

    # Convert pandas timedeltas to hours
    hours = differences.dt.total_seconds() / 3600.0

    # Remove zero/negative values
    hours = hours[hours > 0]

    if hours.empty:
        raise ValueError(
            "No positive inter-event times were found."
        )

    return hours.reset_index(drop=True)


def calculate_interevent_times(catalog) -> np.ndarray:
    """
    Calculate inter-event times in hours.

    Parameters
    ----------
    catalog
        Project Catalog or pandas DataFrame.

    Returns
    -------
    numpy.ndarray
        Inter-event times in hours.
    """
    return _get_interevent_times(catalog).to_numpy()


def plot_interevent_times(
    catalog,
    ax=None,
):
    """
    Plot the distribution of inter-event times.

    A logarithmic x-axis is used because earthquake inter-event
    times can span several orders of magnitude.
    """
    interevent = calculate_interevent_times(catalog)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    # Log-spaced bins
    bins = np.logspace(
        np.log10(interevent.min()),
        np.log10(interevent.max()),
        30,
    )

    ax.hist(
        interevent,
        bins=bins,
        alpha=0.75,
    )

    ax.set_xscale("log")

    ax.set_xlabel("Inter-event time (hours)")
    ax.set_ylabel("Number of intervals")
    ax.set_title("Earthquake Inter-event Time Distribution")

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    return ax


def plot_interevent_loglog(
    catalog,
    ax=None,
):
    """
    Plot the inter-event-time distribution on log-log axes.

    This is useful for inspecting scaling behavior in earthquake
    catalogs and is particularly relevant to temporal clustering.
    """
    interevent = calculate_interevent_times(catalog)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    sorted_times = np.sort(interevent)

    counts = np.arange(
        len(sorted_times),
        0,
        -1,
    )

    ax.loglog(
        sorted_times,
        counts,
        "o",
        markersize=4,
    )

    ax.set_xlabel("Inter-event time (hours)")
    ax.set_ylabel("Number of intervals ≥ Δt")
    ax.set_title("Inter-event Time Survival Distribution")

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    return ax


__all__ = [
    "calculate_interevent_times",
    "plot_interevent_times",
    "plot_interevent_loglog",
]