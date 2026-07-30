from eq_toolkit.sources.comcat import get_events


def test_get_events():
    catalog = get_events(
        starttime="2024-01-01",
        endtime="2024-01-02",
        minmagnitude=6.0,
    )

    assert catalog is not None
    assert len(catalog.data) > 0