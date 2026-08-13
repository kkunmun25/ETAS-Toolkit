import matplotlib.pyplot as plt

from eq_toolkit.catalog.model import Catalog
from eq_toolkit.viz.interevent import (
    plot_interevent_times,
    plot_interevent_loglog,
)


catalog = Catalog()

times = [
    "2026-01-01T00:00:00",
    "2026-01-01T01:00:00",
    "2026-01-01T03:00:00",
    "2026-01-01T07:00:00",
    "2026-01-01T12:00:00",
    "2026-01-01T20:00:00",
    "2026-01-02T06:00:00",
    "2026-01-03T00:00:00",
]

magnitudes = [
    2.0,
    2.5,
    3.0,
    2.2,
    3.5,
    2.8,
    4.0,
    3.2,
]


for i, (time, magnitude) in enumerate(
    zip(times, magnitudes)
):

    catalog.add_event(
        time=time,
        latitude=10.0 + i * 0.1,
        longitude=20.0 + i * 0.1,
        depth=5.0 + i,
        magnitude=magnitude,
        magnitude_type="Mw",
        source_agency="TEST",
        event_id=f"interevent_plot_{i}",
    )


# Plot 1: Inter-event time distribution


plot_interevent_times(catalog)

plt.tight_layout()

plt.savefig(
    "docs/figures/interevent_distribution_test.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# Plot 2: Log-log survival distribution

plot_interevent_loglog(catalog)

plt.tight_layout()

plt.savefig(
    "docs/figures/interevent_loglog_test.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()