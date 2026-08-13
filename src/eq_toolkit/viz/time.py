"""
Time-based earthquake catalog visualizations.

This module provides:
    1. Time-magnitude stem plot
    2. Cumulative event count
    3. Magnitude-time density using hexbin
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _get_dataframe(catalog) -> pd.DataFrame:
    """
    Extract the DataFrame from the project's Catalog object.
    """
    if isinstance(catalog, pd.DataFrame):
        return catalog.copy()

    if hasattr(catalog, "data"):
        if isinstance(catalog.data, pd.DataFrame):
            return catalog.data.copy()

    raise TypeError(
        "catalog must be a pandas DataFrame or a Catalog "
        "containing a pandas DataFrame in .data"
    )


def _prepare_time_data(catalog) -> pd.DataFrame:
    """
    Extract and clean time and magnitude information.
    """
    df = _get_dataframe(catalog)

    required = ["time", "magnitude"]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Catalog is missing required columns: "
            + ", ".join(missing)
        )

    df = df[required].copy()

    # Convert time to pandas datetime
    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce",
        utc=True,
    )

    # Convert magnitude to numeric
    df["magnitude"] = pd.to_numeric(
        df["magnitude"],
        errors="coerce",
    )

    # Remove invalid rows
    df = df.dropna(
        subset=["time", "magnitude"]
    )

    if df.empty:
        raise ValueError(
            "Catalog contains no valid time and magnitude values."
        )

    # Sort chronologically
    df = df.sort_values("time").reset_index(drop=True)

    return df


def plot_time_magnitude(
    catalog,
    ax=None,
):
    """
    Plot earthquake magnitude as a function of time.

    Parameters
    ----------
    catalog
        Project Catalog object or pandas DataFrame.

    ax : matplotlib.axes.Axes, optional
        Existing axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = _prepare_time_data(catalog)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    markerline, stemlines, baseline = ax.stem(
        df["time"],
        df["magnitude"],
        markerfmt="o",
        basefmt=" ",
    )

    # Make the stems visually lighter
    plt.setp(
        stemlines,
        linewidth=0.8,
        alpha=0.6,
    )

    plt.setp(
        markerline,
        markersize=4,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Magnitude")
    ax.set_title("Earthquake Time-Magnitude Distribution")

    ax.grid(
        True,
        alpha=0.25,
    )

    return ax


def plot_cumulative_count(
    catalog,
    ax=None,
):
    """
    Plot cumulative number of earthquakes through time.

    Parameters
    ----------
    catalog
        Project Catalog object or pandas DataFrame.

    ax : matplotlib.axes.Axes, optional
        Existing axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = _prepare_time_data(catalog)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    cumulative_count = range(
        1,
        len(df) + 1,
    )

    ax.step(
        df["time"],
        cumulative_count,
        where="post",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative number of events")
    ax.set_title("Cumulative Earthquake Count")

    ax.grid(
        True,
        alpha=0.25,
    )

    return ax


def plot_magnitude_time_density(
    catalog,
    ax=None,
    gridsize=30,
):
    """
    Plot magnitude-time density using a hexbin plot.

    Parameters
    ----------
    catalog
        Project Catalog object or pandas DataFrame.

    ax : matplotlib.axes.Axes, optional
        Existing axes.

    gridsize : int, default=30
        Number of hexagonal bins.

    Returns
    -------
    matplotlib.axes.Axes
    """
    df = _prepare_time_data(catalog)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # Matplotlib handles datetime values on the x-axis.
    time_numeric= mdates.date2num(df["time"].dt.to_pydatetime())
    hb = ax.hexbin(
        time_numeric,
        df["magnitude"],
        gridsize=gridsize,
        mincnt=1,
    )
    ax.xaxis_date()  # Ensure x-axis is treated as dates

    colorbar = ax.figure.colorbar(
        hb,
        ax=ax,
    )

    colorbar.set_label("Number of earthquakes")

    ax.set_xlabel("Time")
    ax.set_ylabel("Magnitude")
    ax.set_title("Magnitude-Time Density")

    ax.grid(
        True,
        alpha=0.15,
    )

    return ax


def plot_time_analysis(
    catalog,
    axes=None,
):
    """
    Create all three time-based earthquake plots.

    The three panels are:

        1. Time-magnitude stem plot
        2. Cumulative event count
        3. Magnitude-time hexbin density

    Parameters
    ----------
    catalog
        Project Catalog object or pandas DataFrame.

    axes : array-like of matplotlib.axes.Axes, optional
        Three existing axes. If None, a new 3-panel figure is created.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the three plots.

    Notes
    -----
    This function is intended to be used later by the combined
    Phase-3 EDA plotting function.
    """
    if axes is None:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(11, 14),
        )
    else:
        axes = list(axes)

        if len(axes) != 3:
            raise ValueError(
                "axes must contain exactly three matplotlib axes."
            )

        fig = axes[0].figure

    plot_time_magnitude(
        catalog,
        ax=axes[0],
    )

    plot_cumulative_count(
        catalog,
        ax=axes[1],
    )

    plot_magnitude_time_density(
        catalog,
        ax=axes[2],
    )

    fig.tight_layout()

    return fig


__all__ = [
    "plot_time_magnitude",
    "plot_cumulative_count",
    "plot_magnitude_time_density",
    "plot_time_analysis",
]