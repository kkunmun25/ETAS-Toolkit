import pandas as pd

from eq_toolkit.sources.scrape.csn_chile import CSN_Chile


def test_csn_download():
    """Test that the CSN webpage can be downloaded."""

    scraper = CSN_Chile()

    html = scraper.download()

    assert html is not None
    assert len(html) > 0


def test_csn_parse():
    """Test that earthquake data can be parsed."""

    scraper = CSN_Chile()

    html = scraper.download()

    df = scraper.parse(html)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_csn_dataframe():
    """Test standardized CSN dataframe."""

    scraper = CSN_Chile()

    df = scraper.get_dataframe()

    assert isinstance(df, pd.DataFrame)

    required_columns = [
        "time",
        "latitude",
        "longitude",
        "depth",
        "magnitude",
        "magnitude_type",
        "source_agency",
        "event_id",
    ]

    for column in required_columns:
        assert column in df.columns

    assert len(df) > 0


def test_csn_catalog():
    """Test that CSN data can be converted to Catalog."""

    scraper = CSN_Chile()

    catalog = scraper.get_catalog()

    assert catalog is not None