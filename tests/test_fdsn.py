import numpy as np

from eq_toolkit.catalog.model import Catalog
from eq_toolkit.viz.fmd import plot_fmd


def test_plot_fmd_runs():
    catalog = Catalog()

    magnitudes = [
        2.0, 2.1, 2.1, 2.2, 2.2,
        2.3, 2.3, 2.4, 2.4, 2.5,
        2.6, 2.7, 2.8, 2.9, 3.0,
        3.1, 3.2, 3.3, 3.4, 3.5,
    ]

    for i, magnitude in enumerate(magnitudes):
        catalog.add_event(
            time=f"2026-01-01T00:{i:02d}:00",
            latitude=10.0,
            longitude=20.0,
            depth=10.0,
            magnitude=magnitude,
            magnitude_type="Mw",
            source_agency="TEST",
            event_id=f"test_{i}",
        )

    ax = plot_fmd(
        catalog,
        mc=2.2,
    )

    assert ax is not None
    assert ax.get_xlabel() == "Magnitude"
    assert ax.get_ylabel() == "Number of earthquakes"