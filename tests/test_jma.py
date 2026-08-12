from eq_toolkit.sources.scrape.jma import JMA


def make_sample_jma_record():
    """
    Create a synthetic 96-character JMA record.
    """

    line = [" "] * 96

    # Record type
    line[0] = "J"

    # -------------------------------------------------
    # Origin time
    # 2026-08-12 10:30:15.50 JST
    # -------------------------------------------------

    line[1:5] = "2026"
    line[5:7] = "08"
    line[7:9] = "12"
    line[9:11] = "10"
    line[11:13] = "30"
    line[13:17] = "1550"

    # -------------------------------------------------
    # Latitude
    # 35 degrees 30.00 minutes
    # -------------------------------------------------

    line[21:24] = " 35"
    line[24:28] = "3000"

    # -------------------------------------------------
    # Longitude
    # 139 degrees 45.00 minutes
    # -------------------------------------------------

    line[32:36] = " 139"
    line[36:40] = "4500"

    # -------------------------------------------------
    # Depth
    # 10.00 km
    # -------------------------------------------------

    line[44:49] = "1000 "

    # -------------------------------------------------
    # Magnitude
    # 4.5
    # -------------------------------------------------

    line[52:54] = "45"

    # Magnitude type
    line[54] = "J"

    return "".join(line)


def test_jma_class():

    scraper = JMA()

    assert scraper is not None


def test_jma_parser():

    scraper = JMA()

    sample = make_sample_jma_record()

    catalog = scraper.parse_deck(sample)

    assert len(catalog.data) == 1

    event = catalog.data.iloc[0]

    # 10:30:15.50 JST = 01:30:15.50 UTC
    assert event["time"].year == 2026
    assert event["time"].month == 8
    assert event["time"].day == 12
    assert event["time"].hour == 1
    assert event["time"].minute == 30
    assert event["time"].second == 15

    # 35°30.00'
    assert abs(event["latitude"] - 35.5) < 0.0001

    # 139°45.00'
    assert abs(event["longitude"] - 139.75) < 0.0001

    # 10.00 km
    assert abs(event["depth"] - 10.0) < 0.001

    # M4.5
    assert abs(event["magnitude"] - 4.5) < 0.001

    assert event["magnitude_type"] == "J"
    assert event["source_agency"] == "JMA"


def test_jma_get_events():

    scraper = JMA()

    sample = make_sample_jma_record()

    catalog = scraper.get_events(
        deck_text=sample
    )

    assert len(catalog.data) == 1


def test_jma_real_record():

    scraper = JMA()

    # Real record from the downloaded JMA/NIED
    # Arrival Time Data file.
    sample = (
        "J2026081000054562 006 323938 013 1304363 "
        "015  89804806v   711   7269NW KUMAMOTO PREF          6s"
    )

    catalog = scraper.parse_deck(sample)

    assert len(catalog.data) == 1

    event = catalog.data.iloc[0]

    # -------------------------------------------------
    # Time
    #
    # 2026-08-10 00:05:45.62 JST
    # = 2026-08-09 15:05:45.62 UTC
    # -------------------------------------------------

    assert event["time"].year == 2026
    assert event["time"].month == 8
    assert event["time"].day == 9
    assert event["time"].hour == 15
    assert event["time"].minute == 5
    assert event["time"].second == 45

    # -------------------------------------------------
    # Latitude
    #
    # 32° 39.38'
    # -------------------------------------------------

    expected_latitude = (
        32 + 39.38 / 60.0
    )

    assert abs(
        event["latitude"] - expected_latitude
    ) < 0.0001

    # -------------------------------------------------
    # Longitude
    #
    # 130° 43.63'
    # -------------------------------------------------

    expected_longitude = (
        130 + 43.63 / 60.0
    )

    assert abs(
        event["longitude"] - expected_longitude
    ) < 0.0001

    # -------------------------------------------------
    # Depth
    #
    # 898 -> 8.98 km
    # -------------------------------------------------

    assert abs(
        event["depth"] - 8.98
    ) < 0.001

    # -------------------------------------------------
    # Magnitude
    #
    # 06 -> 0.6
    # -------------------------------------------------

    assert abs(
        event["magnitude"] - 0.6
    ) < 0.001

    # -------------------------------------------------
    # Magnitude type
    # -------------------------------------------------

    assert event["magnitude_type"] == "v"

    assert event["source_agency"] == "JMA"


def test_jma_requires_credentials():

    scraper = JMA(
        username=None,
        password=None,
    )

    # Remove credentials from the environment for this test.
    scraper.username = None
    scraper.password = None

    try:
        scraper.download_deck(
            "https://example.com/test"
        )
    except RuntimeError as error:

        assert "JMA credentials are required" in str(
            error
        )

def test_jma_real_file():

    scraper = JMA()

    filename = r"C:\Users\tapas\Downloads\measure_20260810_1.txt"

    with open(
        filename,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        text = file.read()

    catalog = scraper.parse_deck(text)

    print("REAL JMA EVENTS:", len(catalog.data))
    print(catalog.data.head())

    # The real file must contain at least one event.
    assert len(catalog.data) > 0

    # Check standard Catalog columns.
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

    assert list(catalog.data.columns) == expected_columns        