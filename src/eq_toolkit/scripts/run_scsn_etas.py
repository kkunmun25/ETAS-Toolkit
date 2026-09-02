from pathlib import Path

import numpy as np
import pandas as pd

from eq_toolkit.calibrate.em import run_em_restarts
from eq_toolkit.calibrate.mstep import ETASParameters


# ============================================================
# SETTINGS
# ============================================================

CATALOG_FILE = Path("sc-catalog.txt")

# First real-data test only.
MAX_EVENTS = None  # Use all events.

# Your SCSN catalog starts around M2.5.
MC = 2.5

# ETAS reference magnitude.
M0 = MC


# ============================================================
# READ SCSN CATALOG
# ============================================================

rows = []

with CATALOG_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        # Ignore header / HTML markers.
        if (
            not line
            or line.startswith("#")
            or line.startswith("<")
            or line.startswith(">")
        ):
            continue

        parts = line.split()

        # Actual SCSN record has 13 fields.
        if len(parts) < 13:
            continue

        try:

            date = parts[0]
            time = parts[1]

            magnitude = float(parts[4])
            latitude = float(parts[6])
            longitude = float(parts[7])
            depth = float(parts[8])

        except (ValueError, IndexError):
            continue

        rows.append(
            {
                "time": f"{date} {time}",
                "magnitude": magnitude,
                "latitude": latitude,
                "longitude": longitude,
                "depth": depth,
            }
        )


catalog = pd.DataFrame(rows)

print("Events read:", len(catalog))

if catalog.empty:
    raise RuntimeError(
        "No earthquake records were parsed from sc-catalog.txt."
    )


# ============================================================
# CLEAN
# ============================================================

catalog["time"] = pd.to_datetime(
    catalog["time"],
    errors="coerce",
)

catalog = catalog.dropna(
    subset=["time", "magnitude"]
)

catalog = catalog[
    catalog["magnitude"] >= MC
]

catalog = catalog.sort_values(
    "time"
).reset_index(drop=True)

print(
    "Events after Mc filtering:",
    len(catalog),
)


# ============================================================
# FIRST REAL-DATA TEST
# ============================================================


if MAX_EVENTS is not None and len(catalog) > MAX_EVENTS:

    catalog = catalog.iloc[:MAX_EVENTS].copy()


print(
    "Events used for first EM run:",
    len(catalog),
)


# ============================================================
# CONVERT TIME TO DAYS
# ============================================================

time_days = (
    (
        catalog["time"]
        - catalog["time"].iloc[0]
    )
    .dt.total_seconds()
    / 86400.0
).to_numpy(dtype=float)


magnitudes = (
    catalog["magnitude"]
    .to_numpy(dtype=float)
)


# ============================================================
# CATALOG INFORMATION
# ============================================================

duration = (
    time_days[-1]
    - time_days[0]
)

if duration <= 0:
    raise RuntimeError(
        "Catalog duration must be positive."
    )

event_rate = (
    len(time_days)
    / duration
)

print(
    "Duration (days):",
    duration,
)

print(
    "Observed event rate:",
    event_rate,
)


# ============================================================
# INITIAL ETAS PARAMETERS
# ============================================================

initial_parameters = [

    ETASParameters(
        mu=0.5 * event_rate,
        K=0.05,
        alpha=1.0,
        c=0.01,
        p=1.1,
    ),

    ETASParameters(
        mu=0.8 * event_rate,
        K=0.10,
        alpha=0.8,
        c=0.02,
        p=1.3,
    ),
]


# ============================================================
# RUN EM
# ============================================================

print()
print(
    "Starting ETAS EM calibration..."
)

result = run_em_restarts(
    time_days,
    magnitudes,
    initial_parameters,
    m0=M0,
    max_iterations=100,
    tolerance=1e-5,
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("SCSN ETAS CALIBRATION RESULT")
print("=" * 60)

print(
    "Converged:",
    result.converged,
)

print(
    "Iterations:",
    result.iterations,
)

print(
    "Q / log-likelihood:",
    result.log_likelihood,
)

print()
print("Parameters:")

print(
    "mu    =",
    result.parameters.mu,
)

print(
    "K     =",
    result.parameters.K,
)

print(
    "alpha =",
    result.parameters.alpha,
)

print(
    "c     =",
    result.parameters.c,
)

print(
    "p     =",
    result.parameters.p,
)


# ============================================================
# PROBABILITY CHECK
# ============================================================

print()
print(
    "Probability invariant:"
)

check = (
    result.bg
    + result.rho.sum(axis=1)
)

print(
    "max |bg + sum(rho) - 1| =",
    np.max(
        np.abs(check - 1.0)
    ),
)


# ============================================================
# EXPECTED EVENT COUNTS
# ============================================================

print()

print(
    "Expected background events:",
    np.sum(result.bg),
)

print(
    "Expected triggered events:",
    np.sum(result.rho),
)

print()

print(
    "Total expected events:",
    np.sum(result.bg)
    + np.sum(result.rho),
)

print(
    "Observed events:",
    len(catalog),
)