import numpy as np

from eq_toolkit.quality.bvalue_map import spatial_b_value


def test_spatial_b_value():
    rng = np.random.default_rng(42)

    n = 200

    magnitudes = 2.0 + rng.exponential(
        scale=1.0 / np.log(10),
        size=n,
    )

    latitudes = rng.uniform(
        0.0,
        1.0,
        n,
    )

    longitudes = rng.uniform(
        0.0,
        1.0,
        n,
    )

    grid_latitudes = np.array([
        0.25,
        0.5,
        0.75,
    ])

    grid_longitudes = np.array([
        0.25,
        0.5,
        0.75,
    ])

    b_map, sigma_map = spatial_b_value(
        magnitudes,
        latitudes,
        longitudes,
        mc=2.0,
        grid_latitudes=grid_latitudes,
        grid_longitudes=grid_longitudes,
        radius=0.8,
        min_events=20,
        sigma_max=1.0,
    )

    assert b_map.shape == (3, 3)
    assert sigma_map.shape == (3, 3)

    assert np.any(np.isfinite(b_map))
    assert np.any(np.isfinite(sigma_map))