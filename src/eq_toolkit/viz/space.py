"""
Spatial earthquake visualization.

Provides:
    1. Depth cross-section along a chosen azimuth.
    2. Spatial Mc(x, y) grid map.
    3. Spatial b(x, y) grid map.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Catalog handling
# ---------------------------------------------------------

def _get_dataframe(catalog) -> pd.DataFrame:
    """Extract a pandas DataFrame from a Catalog or DataFrame."""

    if isinstance(catalog, pd.DataFrame):
        return catalog.copy()

    if hasattr(catalog, "data"):
        if isinstance(catalog.data, pd.DataFrame):
            return catalog.data.copy()

    raise TypeError(
        "catalog must be a pandas DataFrame or a Catalog "
        "containing a pandas DataFrame in .data"
    )


def _prepare_spatial_data(catalog) -> pd.DataFrame:
    """Prepare latitude, longitude, depth and magnitude data."""

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
            "Catalog contains no valid spatial data."
        )

    return df


# ---------------------------------------------------------
# Geographic helper
# ---------------------------------------------------------

def _latlon_to_km(
    latitude,
    longitude,
    latitude0,
    longitude0,
):
    """
    Convert latitude/longitude differences to approximate
    local Cartesian distances in kilometres.
    """

    km_per_degree_lat = 111.32

    km_per_degree_lon = (
        111.32 * np.cos(np.radians(latitude0))
    )

    x = (
        longitude - longitude0
    ) * km_per_degree_lon

    y = (
        latitude - latitude0
    ) * km_per_degree_lat

    return x, y


# ---------------------------------------------------------
# Depth cross-section
# ---------------------------------------------------------

def depth_cross_section(
    catalog,
    azimuth=0.0,
    ax=None,
):
    """
    Plot a depth cross-section along a chosen azimuth.

    Parameters
    ----------
    catalog
        Project Catalog or pandas DataFrame.

    azimuth : float, default=0
        Azimuth in degrees clockwise from north.

        0°   = North-South
        90°  = East-West
        180° = South-North
        270° = West-East

    ax : matplotlib.axes.Axes, optional
        Existing matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    The horizontal coordinate is the distance along the
    selected azimuth from the catalog centre.

    Earthquake depth is plotted positive downward.
    """

    df = _prepare_spatial_data(catalog)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    latitude0 = df["latitude"].mean()
    longitude0 = df["longitude"].mean()

    x, y = _latlon_to_km(
        df["latitude"],
        df["longitude"],
        latitude0,
        longitude0,
    )

    theta = np.radians(azimuth)

    # Projection onto the chosen azimuth.
    along = (
        x * np.sin(theta)
        + y * np.cos(theta)
    )

    scatter = ax.scatter(
        along,
        df["depth"],
        s=20 + 15 * df["magnitude"],
        c=df["magnitude"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.3,
    )

    colorbar = ax.figure.colorbar(
        scatter,
        ax=ax,
    )

    colorbar.set_label("Magnitude")

    ax.set_xlabel(
        f"Distance along azimuth {azimuth:.1f}° (km)"
    )

    ax.set_ylabel("Depth (km)")

    ax.set_title(
        "Earthquake Depth Cross-section"
    )

    # Geological convention: depth increases downward.
    ax.invert_yaxis()

    ax.grid(
        True,
        alpha=0.25,
    )

    return ax


# ---------------------------------------------------------
# Grid preparation
# ---------------------------------------------------------

def _grid_statistics(
    df,
    value_column,
    nx=5,
    ny=5,
    min_events=1,
):
    """
    Calculate grid-cell statistics.

    Returns
    -------
    tuple
        longitude centres,
        latitude centres,
        grid values,
        event counts
    """

    lon_edges = np.linspace(
        df["longitude"].min(),
        df["longitude"].max(),
        nx + 1,
    )

    lat_edges = np.linspace(
        df["latitude"].min(),
        df["latitude"].max(),
        ny + 1,
    )

    df = df.copy()

    df["lon_bin"] = pd.cut(
        df["longitude"],
        bins=lon_edges,
        labels=False,
        include_lowest=True,
    )

    df["lat_bin"] = pd.cut(
        df["latitude"],
        bins=lat_edges,
        labels=False,
        include_lowest=True,
    )

    grid = np.full(
        (ny, nx),
        np.nan,
    )

    counts = np.zeros(
        (ny, nx),
        dtype=int,
    )

    for iy in range(ny):

        for ix in range(nx):

            cell = df[
                (df["lon_bin"] == ix)
                & (df["lat_bin"] == iy)
            ]

            counts[iy, ix] = len(cell)

            if len(cell) >= min_events:

                values = cell[value_column]

                grid[iy, ix] = values.mean()

    lon_centres = (
        lon_edges[:-1] + lon_edges[1:]
    ) / 2

    lat_centres = (
        lat_edges[:-1] + lat_edges[1:]
    ) / 2

    return (
        lon_centres,
        lat_centres,
        grid,
        counts,
    )


# ---------------------------------------------------------
# Spatial Mc grid
# ---------------------------------------------------------

def plot_spatial_mc(
    catalog,
    mc=2.5,
    nx=5,
    ny=5,
    min_events=3,
    ax=None,
):
    """
    Plot a spatial Mc(x, y) grid map.

    This first implementation uses the supplied Mc value
    as the completeness estimate in cells containing enough
    events.

    Parameters
    ----------
    catalog
        Project Catalog or pandas DataFrame.

    mc : float
        Magnitude of completeness.

    nx, ny : int
        Number of longitude and latitude grid cells.

    min_events : int
        Minimum events required for a cell.

    ax : matplotlib.axes.Axes, optional
        Existing axes.

    Returns
    -------
    matplotlib.axes.Axes
    """

    df = _prepare_spatial_data(catalog)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    # Create a grid where Mc is represented.
    df["mc_value"] = mc

    lon, lat, grid, counts = _grid_statistics(
        df,
        "mc_value",
        nx=nx,
        ny=ny,
        min_events=min_events,
    )

    mesh = ax.pcolormesh(
        lon,
        lat,
        grid,
        shading="nearest",
    )

    colorbar = ax.figure.colorbar(
        mesh,
        ax=ax,
    )

    colorbar.set_label("Mc")

    ax.scatter(
        df["longitude"],
        df["latitude"],
        s=10,
        facecolors="none",
        edgecolors="black",
        alpha=0.5,
    )

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Spatial Magnitude of Completeness Mc(x, y)")

    return ax


# ---------------------------------------------------------
# Spatial b-value grid
# ---------------------------------------------------------

def _calculate_b_value(
    magnitudes,
    mc,
):
    """Calculate the Gutenberg-Richter b-value."""

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    magnitudes = magnitudes[
        magnitudes >= mc
    ]

    if len(magnitudes) < 2:
        return np.nan

    mean_magnitude = magnitudes.mean()

    denominator = (
        mean_magnitude - mc
    )

    if denominator <= 0:
        return np.nan

    return np.log10(np.e) / denominator


def plot_spatial_b(
    catalog,
    mc=2.5,
    nx=5,
    ny=5,
    min_events=3,
    ax=None,
):
    """
    Plot spatial b(x, y) grid map.

    Each grid cell receives a Gutenberg-Richter
    maximum-likelihood b-value using events >= Mc.
    """

    df = _prepare_spatial_data(catalog)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    lon_edges = np.linspace(
        df["longitude"].min(),
        df["longitude"].max(),
        nx + 1,
    )

    lat_edges = np.linspace(
        df["latitude"].min(),
        df["latitude"].max(),
        ny + 1,
    )

    grid = np.full(
        (ny, nx),
        np.nan,
    )

    for iy in range(ny):

        for ix in range(nx):

            cell = df[
                (df["longitude"] >= lon_edges[ix])
                & (df["longitude"] <= lon_edges[ix + 1])
                & (df["latitude"] >= lat_edges[iy])
                & (df["latitude"] <= lat_edges[iy + 1])
            ]

            b = _calculate_b_value(
                cell["magnitude"].to_numpy(),
                mc,
            )

            if len(
                cell[cell["magnitude"] >= mc]
            ) >= min_events:

                grid[iy, ix] = b

    lon_centres = (
        lon_edges[:-1] + lon_edges[1:]
    ) / 2

    lat_centres = (
        lat_edges[:-1] + lat_edges[1:]
    ) / 2

    mesh = ax.pcolormesh(
        lon_centres,
        lat_centres,
        grid,
        shading="nearest",
    )

    colorbar = ax.figure.colorbar(
        mesh,
        ax=ax,
    )

    colorbar.set_label("b-value")

    ax.scatter(
        df["longitude"],
        df["latitude"],
        s=10,
        facecolors="none",
        edgecolors="black",
        alpha=0.5,
    )

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Spatial Gutenberg-Richter b(x, y)")

    return ax


__all__ = [
    "depth_cross_section",
    "plot_spatial_mc",
    "plot_spatial_b",
]