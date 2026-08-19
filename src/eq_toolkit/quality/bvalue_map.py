"""
Spatial b-value estimation.

Calculates b-values on a regular longitude/latitude grid
using earthquakes within a specified radius.
"""

import numpy as np

from .bvalue import aki_utsu_b_value, shi_bolt_sigma


def spatial_b_value(
    magnitudes,
    latitudes,
    longitudes,
    mc,
    grid_latitudes,
    grid_longitudes,
    radius=1.0,
    min_events=50,
    sigma_max=0.2,
):
    """
    Calculate spatially varying b-values.

    Parameters
    ----------
    magnitudes : array-like
        Earthquake magnitudes.
    latitudes, longitudes : array-like
        Earthquake coordinates.
    mc : float
        Magnitude of completeness.
    grid_latitudes, grid_longitudes : array-like
        Coordinates at which b-values are estimated.
    radius : float
        Search radius in degrees.
    min_events : int
        Minimum number of events required.
    sigma_max : float
        Maximum accepted b-value uncertainty.

    Returns
    -------
    b_map : ndarray
        Spatial b-values.
    sigma_map : ndarray
        Standard errors.
    """

    magnitudes = np.asarray(
        magnitudes,
        dtype=float,
    )

    latitudes = np.asarray(
        latitudes,
        dtype=float,
    )

    longitudes = np.asarray(
        longitudes,
        dtype=float,
    )

    if not (
        len(magnitudes)
        == len(latitudes)
        == len(longitudes)
    ):
        raise ValueError(
            "magnitudes, latitudes and longitudes "
            "must have equal length."
        )

    if radius <= 0:
        raise ValueError("radius must be positive.")

    if min_events < 2:
        raise ValueError(
            "min_events must be at least 2."
        )

    grid_latitudes = np.asarray(
        grid_latitudes,
        dtype=float,
    )

    grid_longitudes = np.asarray(
        grid_longitudes,
        dtype=float,
    )

    b_map = np.full(
        (
            len(grid_latitudes),
            len(grid_longitudes),
        ),
        np.nan,
    )

    sigma_map = np.full_like(
        b_map,
        np.nan,
    )

    valid = (
        np.isfinite(magnitudes)
        & np.isfinite(latitudes)
        & np.isfinite(longitudes)
    )

    magnitudes = magnitudes[valid]
    latitudes = latitudes[valid]
    longitudes = longitudes[valid]

    for i, grid_lat in enumerate(grid_latitudes):

        for j, grid_lon in enumerate(grid_longitudes):

            distance = np.sqrt(
                (latitudes - grid_lat) ** 2
                + (longitudes - grid_lon) ** 2
            )

            nearby = distance <= radius

            local_magnitudes = magnitudes[nearby]

            local_magnitudes = local_magnitudes[
                local_magnitudes >= mc
            ]

            if len(local_magnitudes) < min_events:
                continue

            try:
                b = aki_utsu_b_value(
                    local_magnitudes,
                    mc=mc,
                    delta_m=0.1,
                )

                sigma = shi_bolt_sigma(
                    local_magnitudes,
                    mc=mc,
                    delta_m=0.1,
                )

            except ValueError:
                continue

            if (
                np.isfinite(b)
                and np.isfinite(sigma)
                and sigma <= sigma_max
            ):
                b_map[i, j] = b
                sigma_map[i, j] = sigma

    return b_map, sigma_map