import matplotlib.pyplot as plt

from eq_toolkit.catalog.model import Catalog
from eq_toolkit.viz.time import plot_time_analysis


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


plot_time_analysis(catalog)

plt.savefig(
    "docs/figures/time_analysis_test.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()