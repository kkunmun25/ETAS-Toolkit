import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for testing
from eq_toolkit.catalog.model import Catalog
from eq_toolkit.viz.time import (
    plot_time_magnitude,
    plot_cumulative_count,
    plot_magnitude_time_density,
    plot_time_analysis,
)


def make_test_catalog():
    """
    Create a small earthquake Catalog for testing.
    """
    catalog = Catalog()

    events = [
        ("2026-01-01T00:00:00", 2.0),
        ("2026-01-01T01:00:00", 2.5),
        ("2026-01-01T02:00:00", 3.0),
        ("2026-01-01T04:00:00", 2.2),
        ("2026-01-01T06:00:00", 4.0),
        ("2026-01-01T08:00:00", 3.2),
        ("2026-01-01T12:00:00", 2.8),
        ("2026-01-01T18:00:00", 4.5),
    ]

    for i, (time, magnitude) in enumerate(events):
        catalog.add_event(
            time=time,
            latitude=10.0 + i * 0.1,
            longitude=20.0 + i * 0.1,
            depth=5.0 + i,
            magnitude=magnitude,
            magnitude_type="Mw",
            source_agency="TEST",
            event_id=f"time_test_{i}",
        )

    return catalog


def test_plot_time_magnitude_runs():
    catalog = make_test_catalog()

    ax = plot_time_magnitude(catalog)

    assert ax is not None
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Magnitude"


def test_plot_cumulative_count_runs():
    catalog = make_test_catalog()

    ax = plot_cumulative_count(catalog)

    assert ax is not None
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Cumulative number of events"


def test_plot_magnitude_time_density_runs():
    catalog = make_test_catalog()

    ax = plot_magnitude_time_density(catalog)

    assert ax is not None
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Magnitude"


def test_plot_time_analysis_runs():
    catalog = make_test_catalog()

    fig = plot_time_analysis(catalog)

    assert fig is not None
    assert len(fig.axes) >= 3