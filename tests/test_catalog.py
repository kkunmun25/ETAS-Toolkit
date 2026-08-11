from eq_toolkit.catalog.model import Catalog


def test_merge_catalogs():

    cat1 = Catalog()
    cat2 = Catalog()

    cat1.add_event(
        1, 10, 20, 5, 4.5, "Mw", "USGS", "A"
    )

    cat2.add_event(
        2, 11, 21, 6, 5.2, "Mw", "USGS", "B"
    )

    cat1.merge(cat2)

    assert len(cat1.data) == 2