import numpy as np
import pandas as pd

from eq_toolkit.analysis.catalog_compare import (
    compare_event_counts,
    magnitude_count_comparison,
    match_common_events,
    compare_magnitudes,
    compare_locations,
)


def make_catalog(
    times,
    latitudes,
    longitudes,
    magnitudes,
):
    return pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "latitude": latitudes,
            "longitude": longitudes,
            "magnitude": magnitudes,
        }
    )


def test_compare_event_counts():

    catalog_a = make_catalog(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
        ],
        [10, 11, 12],
        [90, 91, 92],
        [3.0, 3.5, 4.0],
    )

    catalog_b = make_catalog(
        [
            "2020-01-01",
            "2020-01-02",
        ],
        [10, 11],
        [90, 91],
        [3.1, 3.6],
    )

    result = compare_event_counts(
        catalog_a,
        catalog_b,
    )

    assert result["catalog_a_count"] == 3
    assert result["catalog_b_count"] == 2
    assert result["count_difference"] == 1


def test_magnitude_count_comparison():

    catalog_a = make_catalog(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
        ],
        [10, 10, 10],
        [90, 90, 90],
        [2.0, 2.0, 3.0],
    )

    catalog_b = make_catalog(
        [
            "2020-01-01",
            "2020-01-02",
        ],
        [10, 10],
        [90, 90],
        [3.0, 3.0],
    )

    result = magnitude_count_comparison(
        catalog_a,
        catalog_b,
        magnitude_bins=np.array(
            [1.5, 2.5, 3.5]
        ),
    )

    assert result["count_a"].sum() == 3
    assert result["count_b"].sum() == 2


def test_match_common_events():

    catalog_a = make_catalog(
        ["2020-01-01 00:00:00"],
        [10.0],
        [90.0],
        [4.0],
    )

    catalog_b = make_catalog(
        ["2020-01-01 00:00:05"],
        [10.01],
        [90.01],
        [4.1],
    )

    result = match_common_events(
        catalog_a,
        catalog_b,
        time_tolerance_seconds=10,
        distance_tolerance_km=20,
    )

    assert len(result) == 1
    assert abs(
        result.iloc[0]["magnitude_difference"]
        + 0.1
    ) < 1e-10


def test_compare_magnitudes():

    matched = pd.DataFrame(
        {
            "magnitude_difference": [
                0.1,
                0.2,
                -0.1,
            ]
        }
    )

    result = compare_magnitudes(
        matched
    )

    assert result["n_common_events"] == 3
    assert result["mae"] > 0


def test_compare_locations():

    matched = pd.DataFrame(
        {
            "distance_km": [
                1.0,
                2.0,
                3.0,
            ]
        }
    )

    result = compare_locations(
        matched
    )

    assert result["n_common_events"] == 3
    assert result["median_distance_km"] == 2.0