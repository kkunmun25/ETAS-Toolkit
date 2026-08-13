import matplotlib

matplotlib.use("Agg")

from eq_toolkit.catalog.model import Catalog
from eq_toolkit.viz.eda import plot_eda


def make_catalog():

    catalog = Catalog()

    events = [
        ("2026-01-01T00:00:00", 10.0, 20.0, 5.0, 2.5),
        ("2026-01-01T01:00:00", 10.1, 20.1, 8.0, 2.8),
        ("2026-01-01T03:00:00", 10.2, 20.2, 10.0, 3.0),
        ("2026-01-01T05:00:00", 10.3, 20.3, 12.0, 3.2),
        ("2026-01-01T08:00:00", 10.4, 20.4, 15.0, 3.5),
        ("2026-01-01T12:00:00", 10.5, 20.5, 18.0, 3.8),
        ("2026-01-01T18:00:00", 10.6, 20.6, 20.0, 4.0),
        ("2026-01-02T06:00:00", 10.7, 20.7, 25.0, 4.5),
    ]

    for i, (
        time,
        latitude,
        longitude,
        depth,
        magnitude,
    ) in enumerate(events):

        catalog.add_event(
            time=time,
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            magnitude=magnitude,
            magnitude_type="Mw",
            source_agency="TEST",
            event_id=f"eda_test_{i}",
        )

    return catalog


def test_plot_eda_runs():

    catalog = make_catalog()

    fig = plot_eda(
        catalog,
        mc=2.5,
        azimuth=45,
    )

    assert fig is not None

    # Six main EDA panels should exist.
    assert len(fig.axes) >= 6