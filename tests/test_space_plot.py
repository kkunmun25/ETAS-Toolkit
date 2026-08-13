import matplotlib.pyplot as plt

from eq_toolkit.catalog.model import Catalog
from eq_toolkit.viz.space import (
    depth_cross_section,
    plot_spatial_mc,
    plot_spatial_b,
)


catalog = Catalog()

events = [
    (10.0, 20.0, 5.0, 2.5),
    (10.2, 20.2, 10.0, 3.0),
    (10.4, 20.4, 15.0, 3.5),
    (10.6, 20.6, 20.0, 4.0),
    (10.8, 20.8, 25.0, 4.5),
    (11.0, 21.0, 30.0, 3.2),
    (11.2, 21.2, 35.0, 3.8),
    (11.4, 21.4, 40.0, 4.2),
    (11.6, 21.6, 45.0, 5.0),
]


for i, (
    latitude,
    longitude,
    depth,
    magnitude,
) in enumerate(events):

    catalog.add_event(
        time=f"2026-01-01T00:{i:02d}:00",
        latitude=latitude,
        longitude=longitude,
        depth=depth,
        magnitude=magnitude,
        magnitude_type="Mw",
        source_agency="TEST",
        event_id=f"space_plot_{i}",
    )


# --------------------------------------------------
# 1. Depth cross-section
# --------------------------------------------------

plot = depth_cross_section(
    catalog,
    azimuth=45,
)

plt.tight_layout()

plt.savefig(
    "docs/figures/depth_cross_section_test.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# --------------------------------------------------
# 2. Spatial Mc
# --------------------------------------------------

plot_spatial_mc(
    catalog,
    mc=2.5,
    nx=3,
    ny=3,
    min_events=1,
)

plt.tight_layout()

plt.savefig(
    "docs/figures/spatial_mc_test.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# --------------------------------------------------
# 3. Spatial b-value
# --------------------------------------------------

plot_spatial_b(
    catalog,
    mc=2.5,
    nx=3,
    ny=3,
    min_events=1,
)

plt.tight_layout()

plt.savefig(
    "docs/figures/spatial_b_test.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()