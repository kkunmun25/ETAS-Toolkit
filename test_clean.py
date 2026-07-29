import pandas as pd

from src.eq_toolkit.catalog.model import Catalog
from src.eq_toolkit.catalog.clean import (
    drop_duplicates,
    sort_by_time,
    filter_magnitude,
)


def test_drop_duplicates():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "latitude": [10, 10, 11],
            "longitude": [20, 20, 21],
            "magnitude": [4.5, 4.5, 5.0],
            "depth": [5, 5, 10],
            "magnitude_type": ["Mw", "Mw", "Mw"],
            "source_agency": ["USGS", "USGS", "USGS"],
            "event_id": ["1", "1", "2"],
})
    

    catalog = Catalog()
    catalog.data = df

    cleaned = drop_duplicates(catalog)

    assert len(cleaned.data) == 2


def test_sort_by_time():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "latitude": [1, 2, 3],
            "longitude": [4, 5, 6],
            "magnitude": [3, 4, 5],
            "depth": [5, 5, 10],
    "magnitude_type": ["Mw", "Mw", "Mw"],
    "source_agency": ["USGS", "USGS", "USGS"],
    "event_id": ["1", "1", "2"],

        }
    )

    catalog = Catalog()
    catalog.data = df

    cleaned = sort_by_time(catalog)

    assert cleaned.data.iloc[0]["time"] == pd.Timestamp("2024-01-01")


def test_filter_magnitude():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "latitude": [1, 2, 3],
            "longitude": [4, 5, 6],
            "magnitude": [3.0, 4.5, 6.0],
            "depth": [5, 5, 10],
    "magnitude_type": ["Mw", "Mw", "Mw"],
    "source_agency": ["USGS", "USGS", "USGS"],
    "event_id": ["1", "1", "2"],

        }
    )

    catalog = Catalog()
    catalog.data = df

    cleaned = filter_magnitude(catalog, min_mag=4)

    assert len(cleaned.data) == 2