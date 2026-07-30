from eq_toolkit.sources.afad import AFAD


def test_afad():

    client = AFAD()

    catalog = client.get_events(
        starttime="2024-01-01",
        endtime="2024-01-10",
        minlatitude=35,
        maxlatitude=43,
        minlongitude=25,
        maxlongitude=45,
        minmagnitude=4.0,
    )

    print(catalog.data.head())

    assert len(catalog.data) > 0