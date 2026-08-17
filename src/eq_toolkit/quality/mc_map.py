"""
Spatial magnitude-of-completeness mapping.

This module estimates spatially varying Mc by sampling a
constant number of nearest earthquakes around every point
on a geographic grid.
"""

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from eq_toolkit.quality.mc import maxc


# =====================================================================
# CONSTANT-N NEAREST-NEIGHBOR SAMPLING
# =====================================================================

def nearest_neighbor_magnitudes(
    catalog,
    latitude,
    longitude,
    n_neighbors=100,
    latitude_col="latitude",
    longitude_col="longitude",
    magnitude_col="magnitude",
):
    """
    Return magnitudes of the N earthquakes nearest to a location.
 
    """

    if not isinstance(
        catalog,
        pd.DataFrame,
    ):
        raise TypeError(
            "catalog must be a pandas DataFrame."
        )

    required_columns = [
        latitude_col,
        longitude_col,
        magnitude_col,
    ]

    missing = [
        column
        for column in required_columns
        if column not in catalog.columns
    ]

    if missing:
        raise ValueError(
            f"Missing catalog columns: {missing}"
        )

    if n_neighbors < 1:
        raise ValueError(
            "n_neighbors must be at least 1."
        )

    # ---------------------------------------------------------------
    # Extract valid geographic coordinates and magnitudes
    # ---------------------------------------------------------------

    data = catalog[
        required_columns
    ].copy()

    data = data.dropna()

    data = data[
        np.isfinite(
            data[latitude_col]
        )
        &
        np.isfinite(
            data[longitude_col]
        )
        &
        np.isfinite(
            data[magnitude_col]
        )
    ]

    if len(data) < n_neighbors:
        raise ValueError(
            "Catalog contains fewer events than "
            "n_neighbors."
        )

    # ---------------------------------------------------------------
    # Convert geographic coordinates to 3-D unit-sphere coordinates.
    #
    # This lets cKDTree calculate great-circle-equivalent nearest
    # neighbors without requiring an additional geographic package.
    # ---------------------------------------------------------------

    lat = np.radians(
        data[latitude_col].to_numpy(
            dtype=float
        )
    )

    lon = np.radians(
        data[longitude_col].to_numpy(
            dtype=float
        )
    )

    x = (
        np.cos(lat)
        * np.cos(lon)
    )

    y = (
        np.cos(lat)
        * np.sin(lon)
    )

    z = np.sin(lat)

    coordinates = np.column_stack(
        [
            x,
            y,
            z,
        ]
    )

    tree = cKDTree(
        coordinates
    )

    # ---------------------------------------------------------------
    # Convert target grid point to 3-D coordinates
    # ---------------------------------------------------------------

    target_lat = np.radians(
        float(latitude)
    )

    target_lon = np.radians(
        float(longitude)
    )

    target = np.array(
        [
            np.cos(target_lat)
            * np.cos(target_lon),

            np.cos(target_lat)
            * np.sin(target_lon),

            np.sin(target_lat),
        ]
    )

    # ---------------------------------------------------------------
    # Find N nearest events
    # ---------------------------------------------------------------

    _, indices = tree.query(
        target,
        k=n_neighbors,
    )

    indices = np.atleast_1d(
        indices
    )

    return data.iloc[
        indices
    ][magnitude_col].to_numpy(
        dtype=float
    )

# =====================================================================
# SPATIAL GRID
# =====================================================================

def create_mc_grid(
    latitude_min,
    latitude_max,
    longitude_min,
    longitude_max,
    latitude_step=0.5,
    longitude_step=0.5,
):
    """
    Create a regular geographic grid for spatial Mc estimation.

    Parameters
    ----------
    latitude_min : float
        Minimum latitude.

    latitude_max : float
        Maximum latitude.

    longitude_min : float
        Minimum longitude.

    longitude_max : float
        Maximum longitude.

    latitude_step : float, default=0.5
        Latitude spacing in degrees.

    longitude_step : float, default=0.5
        Longitude spacing in degrees.

    Returns
    -------
    pandas.DataFrame
        Grid containing:

        latitude
            Grid-point latitude.

        longitude
            Grid-point longitude.
    """

    if latitude_min >= latitude_max:
        raise ValueError(
            "latitude_min must be smaller than latitude_max."
        )

    if longitude_min >= longitude_max:
        raise ValueError(
            "longitude_min must be smaller than longitude_max."
        )

    if latitude_step <= 0:
        raise ValueError(
            "latitude_step must be positive."
        )

    if longitude_step <= 0:
        raise ValueError(
            "longitude_step must be positive."
        )

    latitudes = np.arange(
        latitude_min,
        latitude_max + latitude_step / 2.0,
        latitude_step,
    )

    longitudes = np.arange(
        longitude_min,
        longitude_max + longitude_step / 2.0,
        longitude_step,
    )

    latitude_grid, longitude_grid = np.meshgrid(
        latitudes,
        longitudes,
        indexing="ij",
    )

    grid = pd.DataFrame(
        {
            "latitude": latitude_grid.ravel(),
            "longitude": longitude_grid.ravel(),
        }
    )

    return grid   

# =====================================================================
# SPATIAL Mc
# =====================================================================

def spatial_mc(
    catalog,
    grid,
    n_neighbors=100,
    bin_width=0.1,
    latitude_col="latitude",
    longitude_col="longitude",
    magnitude_col="magnitude",
):
    """
    Estimate spatially varying Mc on a geographic grid.

    A constant number of nearest earthquakes is selected around
    every grid point. Mc is then estimated independently from
    each local sample using the MAXC method.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Earthquake catalog.

    grid : pandas.DataFrame
        Grid containing 'latitude' and 'longitude' columns.

    n_neighbors : int, default=100
        Number of nearest earthquakes used at every grid point.

    bin_width : float, default=0.1
        Magnitude bin width for MAXC.

    latitude_col : str, default="latitude"
        Latitude column in catalog.

    longitude_col : str, default="longitude"
        Longitude column in catalog.

    magnitude_col : str, default="magnitude"
        Magnitude column in catalog.

    Returns
    -------
    pandas.DataFrame
        Grid with an additional 'mc' column.

    Notes
    -----
    The physical search radius is allowed to vary from grid point
    to grid point. The number of earthquakes remains constant.
    """

    if not isinstance(
        grid,
        pd.DataFrame,
    ):
        raise TypeError(
            "grid must be a pandas DataFrame."
        )

    required_grid_columns = [
        "latitude",
        "longitude",
    ]

    missing_grid = [
        column
        for column in required_grid_columns
        if column not in grid.columns
    ]

    if missing_grid:
        raise ValueError(
            f"Missing grid columns: {missing_grid}"
        )

    if n_neighbors < 1:
        raise ValueError(
            "n_neighbors must be at least 1."
        )

    # ---------------------------------------------------------------
    # Prepare catalog
    # ---------------------------------------------------------------

    required_catalog_columns = [
        latitude_col,
        longitude_col,
        magnitude_col,
    ]

    missing_catalog = [
        column
        for column in required_catalog_columns
        if column not in catalog.columns
    ]

    if missing_catalog:
        raise ValueError(
            f"Missing catalog columns: {missing_catalog}"
        )

    data = catalog[
        required_catalog_columns
    ].copy()

    data = data.dropna()

    data = data[
        np.isfinite(
            data[latitude_col]
        )
        &
        np.isfinite(
            data[longitude_col]
        )
        &
        np.isfinite(
            data[magnitude_col]
        )
    ]

    if len(data) < n_neighbors:
        raise ValueError(
            "Catalog contains fewer events than "
            "n_neighbors."
        )

    # ---------------------------------------------------------------
    # Build spherical KD-tree once.
    #
    # We do NOT rebuild the tree for every grid point.
    # ---------------------------------------------------------------

    lat = np.radians(
        data[latitude_col].to_numpy(
            dtype=float
        )
    )

    lon = np.radians(
        data[longitude_col].to_numpy(
            dtype=float
        )
    )

    coordinates = np.column_stack(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )

    tree = cKDTree(
        coordinates
    )

    # ---------------------------------------------------------------
    # Copy grid so original DataFrame isn't modified.
    # ---------------------------------------------------------------

    result = grid.copy()

    mc_values = []

    # ---------------------------------------------------------------
    # Calculate Mc at every grid point.
    # ---------------------------------------------------------------

    for _, point in result.iterrows():

        point_lat = np.radians(
            float(point["latitude"])
        )

        point_lon = np.radians(
            float(point["longitude"])
        )

        target = np.array(
            [
                np.cos(point_lat)
                * np.cos(point_lon),

                np.cos(point_lat)
                * np.sin(point_lon),

                np.sin(point_lat),
            ]
        )

        _, indices = tree.query(
            target,
            k=n_neighbors,
        )

        indices = np.atleast_1d(
            indices
        )

        local_magnitudes = data.iloc[
            indices
        ][magnitude_col].to_numpy(
            dtype=float
        )

        # -----------------------------------------------------------
        # MAXC for local catalog
        # -----------------------------------------------------------

        try:

            local_mc = maxc(
                local_magnitudes,
                bin_width=bin_width,
            )

        except (ValueError, RuntimeError):

            local_mc = np.nan

        mc_values.append(
            local_mc
        )

    result["mc"] = np.asarray(
        mc_values,
        dtype=float,
    )

    return result 

# =====================================================================
# SPATIAL Mc MAP
# =====================================================================

def plot_mc_map(
    spatial_result,
    ax=None,
    cmap="viridis",
    marker_size=50,
):
    """
    Plot spatially varying magnitude of completeness.

    Parameters
    ----------
    spatial_result : pandas.DataFrame
        Output from spatial_mc(). Must contain:
        latitude, longitude, and mc.

    ax : matplotlib.axes.Axes, optional
        Existing axes. If None, a new figure is created.

    cmap : str, default="viridis"
        Matplotlib colormap.

    marker_size : float, default=50
        Size of grid-point markers.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the Mc map.
    """

    import matplotlib.pyplot as plt

    required = [
        "latitude",
        "longitude",
        "mc",
    ]

    missing = [
        column
        for column in required
        if column not in spatial_result.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if ax is None:
        _, ax = plt.subplots(
            figsize=(8, 6)
        )

    valid = spatial_result[
        ["latitude", "longitude", "mc"]
    ].dropna()

    if valid.empty:
        raise ValueError(
            "No valid Mc values available for plotting."
        )

    scatter = ax.scatter(
        valid["longitude"],
        valid["latitude"],
        c=valid["mc"],
        cmap=cmap,
        s=marker_size,
        edgecolors="black",
        linewidths=0.3,
    )

    colorbar = ax.figure.colorbar(
        scatter,
        ax=ax,
    )

    colorbar.set_label(
        "Magnitude of Completeness (Mc)"
    )

    ax.set_xlabel(
        "Longitude (°)"
    )

    ax.set_ylabel(
        "Latitude (°)"
    )

    ax.set_title(
        "Spatial Magnitude of Completeness"
    )

    return ax