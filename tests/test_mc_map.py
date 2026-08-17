import numpy as np
import pandas as pd

from eq_toolkit.quality.mc_map import (
    nearest_neighbor_magnitudes,
    create_mc_grid,
    spatial_mc ,
    plot_mc_map
)
# CONSTANT-N NEAREST-NEIGHBOR TEST

def test_nearest_neighbor_magnitudes():

    catalog = pd.DataFrame(
        {
            "latitude": [
                10.00,
                10.01,
                10.02,
                10.03,
                10.04,
                10.05,
                20.00,
                20.01,
            ],
            "longitude": [
                80.00,
                80.01,
                80.02,
                80.03,
                80.04,
                80.05,
                90.00,
                90.01,
            ],
            "magnitude": [
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
                5.0,
                5.1,
            ],
        }
    )

    magnitudes = nearest_neighbor_magnitudes(
        catalog,
        latitude=10.02,
        longitude=80.02,
        n_neighbors=5,
    )

    assert len(magnitudes) == 5

    assert np.all(
        np.isfinite(magnitudes)
    )

    assert np.max(
        magnitudes
    ) < 5.0

# SPATIAL GRID TEST

def test_create_mc_grid():

    grid = create_mc_grid(
        latitude_min=10.0,
        latitude_max=11.0,
        longitude_min=80.0,
        longitude_max=81.0,
        latitude_step=0.5,
        longitude_step=0.5,
    )

    

    assert len(grid) == 9

    assert "latitude" in grid.columns
    assert "longitude" in grid.columns

    assert np.isfinite(
        grid["latitude"]
    ).all()

    assert np.isfinite(
        grid["longitude"]
    ).all()

    assert grid["latitude"].min() == 10.0
    assert grid["latitude"].max() == 11.0

    assert grid["longitude"].min() == 80.0
    assert grid["longitude"].max() == 81.0    

# SPATIAL Mc TEST

def test_spatial_mc():

    rng = np.random.default_rng(42)

    # Synthetic earthquake catalog

    n_events = 500

    catalog = pd.DataFrame(
        {
            "latitude": rng.uniform(
                10.0,
                11.0,
                n_events,
            ),

            "longitude": rng.uniform(
                80.0,
                81.0,
                n_events,
            ),

            "magnitude": np.round(
                1.5
                + rng.exponential(
                    scale=0.5,
                    size=n_events,
                ),
                1,
            ),
        }
    )

    # Create grid

    grid = create_mc_grid(
        latitude_min=10.0,
        latitude_max=11.0,
        longitude_min=80.0,
        longitude_max=81.0,
        latitude_step=0.5,
        longitude_step=0.5,
    )

    # Calculate spatial Mc

    result = spatial_mc(
        catalog,
        grid,
        n_neighbors=100,
        bin_width=0.1,
    )
    # Check structure
    

    assert len(result) == len(grid)

    assert "latitude" in result.columns
    assert "longitude" in result.columns
    assert "mc" in result.columns

    # At least some grid points should have a valid Mc.


    assert result["mc"].notna().any()

    valid_mc = result["mc"].dropna()

    assert np.all(
        np.isfinite(valid_mc)
    )

    # Mc should lie inside the magnitude range.
    assert (
        valid_mc.min()
        >= catalog["magnitude"].min()
    )

    assert (
        valid_mc.max()
        <= catalog["magnitude"].max()
    )   


# SPATIAL Mc MAP TEST


def test_plot_mc_map():

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    spatial_result = pd.DataFrame(
        {
            "latitude": [
                10.0,
                10.5,
                11.0,
                10.0,
                10.5,
                11.0,
            ],
            "longitude": [
                80.0,
                80.0,
                80.0,
                80.5,
                80.5,
                80.5,
            ],
            "mc": [
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
                2.6,
            ],
        }
    )

    ax = plot_mc_map(
        spatial_result
    )

    assert ax is not None

    assert ax.get_xlabel() == "Longitude (°)"

    assert ax.get_ylabel() == "Latitude (°)"

    assert (
        ax.get_title()
        == "Spatial Magnitude of Completeness"
    )

    plt.close(
        ax.figure
    )     