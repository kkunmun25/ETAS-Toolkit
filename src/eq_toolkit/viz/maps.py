"""
Earthquake epicenter visualization.

This module provides a simple earthquake epicenter map using
longitude and latitude.

Marker size is proportional to earthquake magnitude.
Marker color represents earthquake depth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def _get_required_columns(catalog) -> pd.DataFrame:
    """
    Return a clean DataFrame containing the fields required
    for the epicenter map.
    """
    df = _get_dataframe(catalog)

    required = [
        "latitude",
        "longitude",
        "depth",
        "magnitude",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Catalog is missing required columns: "
            + ", ".join(missing)
        )

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()

    if df.empty:
        raise ValueError(
            "Catalog contains no valid latitude, longitude, "
            "depth, and magnitude values."
        )

    return df


def _marker_sizes(
    magnitudes: np.ndarray,
    scale: float = 20.0,
) -> np.ndarray:
    """
    Convert earthquake magnitude into marker sizes.

    A logarithmic magnitude scale is used so that large earthquakes
    are visibly larger without making the markers excessively large.
    """
    minimum = np.nanmin(magnitudes)

    sizes = scale * (10 ** (0.6 * (magnitudes - minimum)))

    return sizes


def plot_epicenters(
    catalog,
    ax=None,
    size_scale: float = 20.0,
):
    """
    Plot earthquake epicenters.

    Parameters
    ----------
    catalog
        Project Catalog object or pandas DataFrame.

    ax : matplotlib.axes.Axes, optional
        Existing axes. If None, a new figure and axes are created.

    size_scale : float, default=20
        Controls the overall marker size.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the earthquake epicenter map.

    Notes
    -----
    Longitude is plotted on the x-axis.
    Latitude is plotted on the y-axis.

    Marker size is proportional to earthquake magnitude.

    Marker color represents earthquake depth.
    """
    df = _get_required_columns(catalog)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    magnitudes = df["magnitude"].to_numpy()
    depths = df["depth"].to_numpy()

    sizes = _marker_sizes(
        magnitudes,
        scale=size_scale,
    )

    scatter = ax.scatter(
        df["longitude"],
        df["latitude"],
        s=sizes,
        c=depths,
        cmap="viridis",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.3,
    )

    colorbar = ax.figure.colorbar(
        scatter,
        ax=ax,
    )

    colorbar.set_label("Depth (km)")

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Earthquake Epicenters")

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.set_aspect("equal", adjustable="datalim")

    return ax


__all__ = [
    "plot_epicenters",
]