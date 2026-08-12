from eq_toolkit.sources.scrape.cwa_taiwan import CWA_Taiwan


def test_cwa_catalog():

    scraper = CWA_Taiwan()

    catalog = scraper.get_events()

    assert catalog is not None
    assert len(catalog.data) > 0

    assert "time" in catalog.data.columns
    assert "latitude" in catalog.data.columns
    assert "longitude" in catalog.data.columns
    assert "depth" in catalog.data.columns
    assert "magnitude" in catalog.data.columns
    assert "magnitude_type" in catalog.data.columns
    assert "source_agency" in catalog.data.columns
    assert "event_id" in catalog.data.columns