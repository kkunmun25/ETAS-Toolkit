import matplotlib.pyplot as plt

from eq_toolkit.catalog.model import Catalog
from eq_toolkit.viz.maps import plot_epicenters


catalog = Catalog()

events = [
    (10.0, 20.0, 5.0, 2.5),
    (10.5, 20.5, 10.0, 3.0),
    (11.0, 21.0, 20.0, 4.0),
    (11.5, 21.5, 30.0, 5.0),
    (12.0, 22.0, 50.0, 3.5),
]

for i, (latitude, longitude, depth, magnitude) in enumerate(events):

    catalog.add_event(
        time=f"2026-01-01T00:{i:02d}:00",
        latitude=latitude,
        longitude=longitude,
        depth=depth,
        magnitude=magnitude,
        magnitude_type="Mw",
        source_agency="TEST",
        event_id=f"map_test_{i}",
    )


plot_epicenters(catalog)

plt.tight_layout()

plt.savefig(
    "docs/figures/epicenter_test.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()