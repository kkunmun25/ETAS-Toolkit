import matplotlib

matplotlib.use("Agg")

import numpy as np

from eq_toolkit.catalog.model import Catalog
from eq_toolkit.viz.interevent import (
    calculate_interevent_times,
    plot_interevent_times,
    plot_interevent_loglog,
)


def make_catalog():
    catalog = Catalog()

    times = [
        "2026-01-01T00:00:00",
        "2026-01-01T01:00:00",
        "2026-01-01T03:00:00",
        "2026-01-01T07:00:00",
        "2026-01-01T12:00:00",
    ]

    for i, time in enumerate(times):
        catalog.add_event(
            time=time,
            latitude=10.0,
            longitude=20.0,
            depth=10.0,
            magnitude=2.5 + i * 0.2,
            magnitude_type="Mw",
            source_agency="TEST",
            event_id=f"interevent_{i}",
        )

    return catalog


def test_calculate_interevent_times():
    catalog = make_catalog()

    values = calculate_interevent_times(catalog)

    expected = np.array([
        1.0,
        2.0,
        4.0,
        5.0,
    ])

    assert np.allclose(values, expected)


def test_plot_interevent_times():
    catalog = make_catalog()

    ax = plot_interevent_times(catalog)

    assert ax is not None
    assert ax.get_xlabel() == "Inter-event time (hours)"


def test_plot_interevent_loglog():
    catalog = make_catalog()

    ax = plot_interevent_loglog(catalog)

    assert ax is not None
    assert ax.get_xlabel() == "Inter-event time (hours)"