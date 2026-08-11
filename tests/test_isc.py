from eq_toolkit.sources.isc import get_events


def test_isc_download():
    catalog = get_events(
        starttime="2024-01-01",
        endtime="2024-01-02",
        minmagnitude=5.0,
    )

    

