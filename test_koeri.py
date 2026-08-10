from eq_toolkit.sources.scrape.koeri import KOERI


def test_koeri():
    scraper = KOERI()

    catalog = scraper.get_events()

    assert catalog is not None