from eq_toolkit.catalog.model import Catalog
from eq_toolkit.sources.scrape.auth_greece import AUTHGreece


def test_auth_download():

    scraper = AUTHGreece()

    text = scraper.get_raw_data()

    assert text


def test_auth_parse_line():

    scraper = AUTHGreece()

    result = scraper.parse_line(
        "? -550 0000000000.00 36.9000 22.4000 0. 6.8"
    )

    assert result is not None
    assert result["year"] == -550
    assert result["latitude"] == 36.9
    assert result["longitude"] == 22.4
    assert result["depth"] == 0.0
    assert result["magnitude"] == 6.8


def test_auth_catalog():

    scraper = AUTHGreece()

    catalog = scraper.get_events(
        min_mag=5.0
    )

    assert isinstance(catalog, Catalog)

    assert "time" in catalog.data.columns
    assert "latitude" in catalog.data.columns
    assert "longitude" in catalog.data.columns
    assert "depth" in catalog.data.columns
    assert "magnitude" in catalog.data.columns
    assert "magnitude_type" in catalog.data.columns
    assert "source_agency" in catalog.data.columns
    assert "event_id" in catalog.data.columns