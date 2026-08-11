from eq_toolkit.sources.scrape.ingv_horus import INGVHORUS


def test_ingv_horus_class():
    scraper = INGVHORUS()

    assert scraper is not None

    assert scraper.CATALOG_URL == (
        "https://horus.bo.ingv.it/"
        "DataFolder/HORUS_Ita_Catalog.zip"
    )