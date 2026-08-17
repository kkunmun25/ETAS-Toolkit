import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eq_toolkit.quality.mc_map import (
    create_mc_grid,
    spatial_mc,
    plot_mc_map,
)


# ---------------------------------------------------------------
# 1. Create example earthquake catalog
# ---------------------------------------------------------------

rng = np.random.default_rng(42)

n_events = 1000

catalog = pd.DataFrame(
    {
        "latitude": rng.uniform(
            10.0,
            11.0,
            n_events,
        ),

        "longitude": rng.uniform(
            80.0,
            81.0,
            n_events,
        ),

        "magnitude": np.round(
            1.5
            + rng.exponential(
                scale=0.5,
                size=n_events,
            ),
            1,
        ),
    }
)


# ---------------------------------------------------------------
# 2. Create spatial grid
# ---------------------------------------------------------------

grid = create_mc_grid(
    latitude_min=10.0,
    latitude_max=11.0,
    longitude_min=80.0,
    longitude_max=81.0,
    latitude_step=0.1,
    longitude_step=0.1,
)


# ---------------------------------------------------------------
# 3. Calculate local Mc
# ---------------------------------------------------------------

result = spatial_mc(
    catalog,
    grid,
    n_neighbors=100,
    bin_width=0.1,
)


# ---------------------------------------------------------------
# 4. Plot Mc map
# ---------------------------------------------------------------

ax = plot_mc_map(
    result
)

ax.set_title(
    "Example Spatial Mc Map"
)


# ---------------------------------------------------------------
# 5. Save figure
# ---------------------------------------------------------------

output_path = (
    "docs/figures/"
    "spatial_mc_example.png"
)

ax.figure.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(
    f"Saved spatial Mc map to: {output_path}"
)

print()
print(result.head())