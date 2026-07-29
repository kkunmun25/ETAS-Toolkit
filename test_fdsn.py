from eq_toolkit.sources.fdsn import FDSNClient


def test_create_client():
    client = FDSNClient(
        "https://earthquake.usgs.gov/fdsnws/event/1"
    )

    assert client.base_url == \
        "https://earthquake.usgs.gov/fdsnws/event/1"


def test_has_get_events():

    client = FDSNClient(
        "https://earthquake.usgs.gov/fdsnws/event/1"
    )

    assert callable(client.get_events)    

from eq_toolkit.catalog.model import Catalog


def test_get_events_returns_catalog():

    client = FDSNClient(
        "https://earthquake.usgs.gov/fdsnws/event/1"
    )

    catalog = client.get_events(
        bbox=[68, 6, 98, 37],
        time_range=("2025-01-01", "2025-01-02"),
        min_mag=6,
    )

    assert isinstance(catalog, Catalog)    