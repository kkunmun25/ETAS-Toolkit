from eq_toolkit.sources.geo_net import GeoNet


def test_geonet_download():
    """
    Test downloading a small earthquake catalogue
    from the GeoNet WFS service.
    """

    source = GeoNet()

    catalog = source.get_events(
        bbox=(170, 180, -48, -34),
        time_range=("2024-01-01", "2024-01-02"),
        min_mag=3.0,
    )

    assert catalog is not None
    assert len(catalog.data) > 0

    expected_columns = [
        "time",
        "latitude",
        "longitude",
        "depth",
        "magnitude",
        "magnitude_type",
        "source_agency",
        "event_id",
    ]

    for column in expected_columns:
        assert column in catalog.data.columns

    assert (catalog.data["magnitude"] >= 3.0).all()

    assert (catalog.data["source_agency"] == "GeoNet").all()