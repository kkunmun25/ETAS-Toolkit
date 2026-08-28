"""

Primary catalog:
    sc-catalog.txt  (SCSN / SCEDC)

Comparison catalog:
    USGS catalog automatically downloaded for the SAME
    geographic region and SAME time period.

Run from repository root:

    python professor_figures.py
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# ============================================================
# SETTINGS
# ============================================================

SC_FILE = Path("sc-catalog.txt")
USGS_FILE = Path("usgs_sc_same_region.csv")

OUT = Path("docs/figures/professor")
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

# ============================================================
# 1. READ SCSN FIXED-WIDTH CATALOG
# ============================================================

def read_scsn(path):

    rows = []

    with open(path, "r", errors="ignore") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line.startswith("<"):
                continue

            parts = line.split()

            # Expected:
            # date time ET GT MAG M LAT LON DEPTH Q EVID ...

            if len(parts) < 10:
                continue

            try:

                date = parts[0]
                time = parts[1]

                dt = pd.to_datetime(
                    date + " " + time,
                    errors="coerce",
                    utc=True
                )

                mag = float(parts[4])
                lat = float(parts[6])
                lon = float(parts[7])
                depth = float(parts[8])
                evid = parts[10]

                if pd.isna(dt):
                    continue

                rows.append(
                    [
                        dt,
                        mag,
                        lat,
                        lon,
                        depth,
                        evid
                    ]
                )

            except Exception:
                continue

    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "magnitude",
            "latitude",
            "longitude",
            "depth",
            "event_id"
        ]
    )

    return df.sort_values("time").reset_index(drop=True)


# ============================================================
# 2. DOWNLOAD USGS CATALOG
# ============================================================

def get_usgs_catalog(sc):

    start = sc.time.min().strftime("%Y-%m-%d")
    end = (
        sc.time.max() + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    min_lat = sc.latitude.min()
    max_lat = sc.latitude.max()

    min_lon = sc.longitude.min()
    max_lon = sc.longitude.max()

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "geojson",
        "starttime": start,
        "endtime": end,
        "minlatitude": float(min_lat),
        "maxlatitude": float(max_lat),
        "minlongitude": float(min_lon),
        "maxlongitude": float(max_lon),
        "minmagnitude": 2.5,
        "orderby": "time-asc",
        "limit": 20000
    }

    print("\nDownloading USGS comparison catalog...")
    print("Region:")
    print(min_lat, max_lat, min_lon, max_lon)
    print("Period:")
    print(start, "to", end)

    r = requests.get(
        url,
        params=params,
        timeout=120
    )

    r.raise_for_status()

    data = r.json()

    rows = []

    for feature in data["features"]:

        prop = feature["properties"]
        geo = feature["geometry"]

        if geo["coordinates"] is None:
            continue

        lon, lat, depth = geo["coordinates"]

        if prop["mag"] is None:
            continue

        rows.append(
            [
                pd.to_datetime(
                    prop["time"],
                    unit="ms",
                    utc=True
                ),
                float(prop["mag"]),
                float(lat),
                float(lon),
                float(depth) if depth is not None else np.nan,
                feature["id"]
            ]
        )

    usgs = pd.DataFrame(
        rows,
        columns=[
            "time",
            "magnitude",
            "latitude",
            "longitude",
            "depth",
            "event_id"
        ]
    )

    usgs = usgs.sort_values("time").reset_index(drop=True)

    usgs.to_csv(USGS_FILE, index=False)

    print(
        f"USGS catalog downloaded: {len(usgs)} events"
    )

    return usgs


# ============================================================
# 3. COMMON HELPERS
# ============================================================

def savefig(fig, number, title):

    fig.suptitle(
        f"Figure {number}: {title}",
        fontsize=13
    )

    fig.tight_layout()

    filename = OUT / f"fig{number:02d}.png"

    fig.savefig(
        filename,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Created:", filename)


def maxc(mags, bin_width=0.1):

    mags = np.asarray(mags, dtype=float)
    mags = mags[np.isfinite(mags)]

    bins = np.arange(
        np.floor(mags.min() / bin_width) * bin_width,
        np.ceil(mags.max() / bin_width) * bin_width
        + bin_width,
        bin_width
    )

    hist, edges = np.histogram(
        mags,
        bins=bins
    )

    return float(
        edges[np.argmax(hist)]
    )


def aki_b(mags, mc, dm=0.1):

    mags = np.asarray(mags, dtype=float)

    x = mags[mags >= mc]

    if len(x) < 5:
        return np.nan

    return (
        np.log(10)
        / (np.mean(x) - mc + dm / 2)
    )


def haversine_km(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2 * R * np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# LOAD CATALOGS
# ============================================================

print("\nReading SCSN catalog...")

sc = read_scsn(SC_FILE)

print(
    f"SCSN events: {len(sc)}"
)

print(
    "SCSN period:",
    sc.time.min(),
    "to",
    sc.time.max()
)

if USGS_FILE.exists():

    usgs = pd.read_csv(
        USGS_FILE,
        parse_dates=["time"]
    )

    usgs["time"] = pd.to_datetime(
        usgs["time"],
        utc=True,
        format="mixed",
    )

else:

    usgs = get_usgs_catalog(sc)


# ============================================================
# FIGURE 1
# EPICENTRE MAP
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

size = (
    8
    * 10 ** (
        0.55
        * (
            sc.magnitude
            - sc.magnitude.min()
        )
    )
)

s = ax.scatter(
    sc.longitude,
    sc.latitude,
    s=size,
    c=sc.magnitude,
    cmap="viridis",
    alpha=0.70,
    edgecolors="black",
    linewidths=0.2
)

fig.colorbar(
    s,
    ax=ax,
    label="Magnitude"
)

ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.grid(alpha=0.25)

savefig(
    fig,
    1,
    "SCSN epicentre map, magnitude-scaled"
)


# ============================================================
# FIGURE 2
# TIME-MAGNITUDE STEM
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))

ax.stem(
    sc.time,
    sc.magnitude,
    markerfmt=".",
    basefmt=" "
)

ax.set_xlabel("Time")
ax.set_ylabel("Magnitude")
ax.grid(alpha=0.25)

savefig(
    fig,
    2,
    "Time-magnitude stem plot"
)


# ============================================================
# FIGURE 3
# TWO CATALOGS — CUMULATIVE COUNT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.step(
    sc.time,
    np.arange(1, len(sc) + 1),
    where="post",
    label="SCSN / SCEDC"
)

ax.step(
    usgs.time,
    np.arange(1, len(usgs) + 1),
    where="post",
    label="USGS"
)

ax.set_xlabel("Time")
ax.set_ylabel("Cumulative event count")
ax.legend()
ax.grid(alpha=0.25)

savefig(
    fig,
    3,
    "Same region and period: cumulative event count"
)


# ============================================================
# MATCH COMMON EVENTS
# ============================================================

def match_catalogs(a, b, tolerance_seconds=60):

    aa = a.sort_values("time").copy()
    bb = b.sort_values("time").copy()

    # Force identical datetime dtype/precision
    aa["time"] = pd.to_datetime(
        aa["time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    bb["time"] = pd.to_datetime(
        bb["time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    matches = pd.merge_asof(
        aa,
        bb,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(
            seconds=tolerance_seconds
        ),
        suffixes=("_scsn", "_usgs")
    )

    matches = matches.dropna(
        subset=[
            "magnitude_scsn",
            "magnitude_usgs"
        ]
    )

    return matches

common = match_catalogs(sc,usgs,tolerance_seconds=60)

print("Common events matched:",len(common))

if len(common) == 0:
    raise RuntimeError(
        "No common events found between SCSN and USGS."
    )


# ============================================================
# FIGURE 4
# COMMON EVENT MAGNITUDE COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    common.magnitude_scsn,
    common.magnitude_usgs,
    s=15,
    alpha=0.5
)

lo = min(
    common.magnitude_scsn.min(),
    common.magnitude_usgs.min()
)

hi = max(
    common.magnitude_scsn.max(),
    common.magnitude_usgs.max()
)

ax.plot(
    [lo, hi],
    [lo, hi],
    "k--",
    label="1:1"
)

ax.set_xlabel("SCSN magnitude")
ax.set_ylabel("USGS magnitude")

ax.legend()
ax.grid(alpha=0.25)

savefig(
    fig,
    4,
    "Magnitude comparison for common events"
)


# ============================================================
# FIGURE 5
# LOCATION DIFFERENCE
# ============================================================

lat_diff = (
    common.latitude_scsn
    - common.latitude_usgs
)

lon_diff = (
    common.longitude_scsn
    - common.longitude_usgs
)

distance = haversine_km(
    common.latitude_scsn,
    common.longitude_scsn,
    common.latitude_usgs,
    common.longitude_usgs
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(
    distance,
    bins=40,
    edgecolor="black"
)

ax.set_xlabel(
    "Epicentre separation (km)"
)

ax.set_ylabel(
    "Number of common events"
)

ax.grid(alpha=0.25)

savefig(
    fig,
    5,
    "Location difference for common events"
)


# ============================================================
# FIGURE 6
# FINE MAGNITUDE HISTOGRAM
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

bins = np.arange(
    np.floor(sc.magnitude.min() * 100) / 100,
    np.ceil(sc.magnitude.max() * 100) / 100
    + 0.01,
    0.01
)

ax.hist(
    sc.magnitude,
    bins=bins,
    edgecolor="black",
    linewidth=0.25
)

ax.set_xlabel("Magnitude")
ax.set_ylabel("Count")

ax.grid(alpha=0.2)

savefig(
    fig,
    6,
    "Fine-binned magnitude histogram"
)


# ============================================================
# FIGURE 7
# REPORTING ARTIFACT — DAILY RATE
# ============================================================

daily = (
    sc
    .set_index("time")
    .resample("D")
    .size()
)

fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(
    daily.index,
    daily.values,
    linewidth=0.8
)

ax.set_xlabel("Date")
ax.set_ylabel("Events per day")

ax.grid(alpha=0.25)

savefig(
    fig,
    7,
    "Catalog reporting-rate diagnostic"
)


# ============================================================
# FIGURE 8
# FMD
# ============================================================

m = sc.magnitude.to_numpy()

mc = maxc(m)

values = np.arange(
    np.floor(m.min() / 0.1) * 0.1,
    np.ceil(m.max() / 0.1) * 0.1 + 0.1,
    0.1
)

incremental = []

cumulative = []

for v in values:

    incremental.append(
        np.sum(
            (
                m >= v - 0.05
            )
            &
            (
                m < v + 0.05
            )
        )
    )

    cumulative.append(
        np.sum(m >= v)
    )

incremental = np.asarray(incremental)
cumulative = np.asarray(cumulative)

fig, ax = plt.subplots(figsize=(8, 6))

mask1 = incremental > 0
mask2 = cumulative > 0

ax.semilogy(
    values[mask1],
    incremental[mask1],
    "o",
    label="Non-cumulative"
)

ax.semilogy(
    values[mask2],
    cumulative[mask2],
    "s-",
    label="Cumulative"
)

ax.axvline(
    mc,
    linestyle="--",
    label=f"Mc = {mc:.2f}"
)

ax.set_xlabel("Magnitude")
ax.set_ylabel("Number of events")

ax.legend()
ax.grid(
    alpha=0.25,
    which="both"
)

savefig(
    fig,
    8,
    "Frequency-magnitude distribution"
)


# ============================================================
# FIGURE 9
# Mc BY MULTIPLE METHODS
# ============================================================

methods = {}

methods["MAXC"] = mc

try:

    from eq_toolkit.quality.mc import (
        gft,
        mbs,
        emr
    )

    for name, func in [
        ("GFT", gft),
        ("MBS", mbs),
        ("EMR", emr)
    ]:

        try:

            result = func(m)

            if isinstance(
                result,
                tuple
            ):
                result = result[0]

            if np.isscalar(result):

                methods[name] = float(
                    result
                )

        except Exception as e:

            print(
                f"{name} failed:",
                e
            )

except Exception as e:

    print(
        "Mc methods import failed:",
        e
    )


fig, ax = plt.subplots(
    figsize=(9, 4)
)

names = list(methods.keys())
vals = list(methods.values())

ax.scatter(
    vals,
    np.zeros(len(vals)),
    s=100
)

for name, value in methods.items():

    ax.annotate(
        f"{name}\n{value:.2f}",
        (value, 0),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center"
    )

ax.set_xlabel(
    "Magnitude of completeness, Mc"
)

ax.set_yticks([])

ax.grid(
    axis="x",
    alpha=0.25
)

savefig(
    fig,
    9,
    "Mc by several methods"
)


# ============================================================
# FIGURE 10
# b VS Mc
# ============================================================

mc_values = np.arange(
    max(
        1.5,
        np.floor(mc * 10) / 10
    ),
    np.floor(
        m.max() * 10
    ) / 10 + 0.1,
    0.1
)

b_values = []
b_sigma = []

for q in mc_values:

    x = m[m >= q]

    if len(x) >= 20:

        b = aki_b(x, q)

        sigma = (
            2.3
            * b
            * b
            * np.std(
                x,
                ddof=1
            )
            / np.sqrt(
                len(x)
                * (len(x) - 1)
            )
        )

        b_values.append(b)
        b_sigma.append(sigma)

    else:

        b_values.append(np.nan)
        b_sigma.append(np.nan)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.errorbar(
    mc_values,
    b_values,
    yerr=b_sigma,
    fmt="o-",
    capsize=3
)

ax.axvline(
    mc,
    linestyle="--",
    label=f"Selected Mc={mc:.2f}"
)

ax.set_xlabel("Mc threshold")
ax.set_ylabel("b-value")

ax.legend()
ax.grid(alpha=0.25)

savefig(
    fig,
    10,
    "b-value versus Mc"
)


# ============================================================
# FIGURE 11
# Mc THROUGH AFTERSHOCK SEQUENCE
# ============================================================

mainshock_index = sc.magnitude.idxmax()

mainshock = sc.loc[
    mainshock_index
]

t0 = mainshock.time

after = sc[
    (
        sc.time >= t0
    )
    &
    (
        sc.time <=
        t0 + pd.Timedelta(days=60)
    )
].copy()

print(
    "\nLargest-event aftershock window:",
    len(after),
    "events"
)

window = 50

times11 = []
mc11 = []

for i in range(
    window,
    len(after),
    max(10, window // 2)
):

    subset = after.iloc[
        i-window:i
    ]

    if len(subset) >= 20:

        times11.append(
            subset.time.iloc[-1]
        )

        mc11.append(
            maxc(
                subset.magnitude
            )
        )


fig, ax = plt.subplots(
    figsize=(10, 5)
)

if len(times11) > 0:

    ax.plot(
        times11,
        mc11,
        "o-"
    )

ax.axvline(
    t0,
    linestyle="--",
    label="Mainshock"
)

ax.set_xlabel("Time")
ax.set_ylabel("Mc")

ax.legend()
ax.grid(alpha=0.25)

savefig(
    fig,
    11,
    "Mc through an aftershock sequence"
)


# ============================================================
# FIGURE 12
# SPATIAL Mc
# ============================================================

sp = sc.dropna(
    subset=[
        "latitude",
        "longitude",
        "magnitude"
    ]
).copy()

# constant-N cells
N_PER_CELL = 100

sp = sp.sort_values(
    "longitude"
)

# spatial grid using quantiles
sp["xbin"] = pd.qcut(
    sp.longitude,
    5,
    labels=False,
    duplicates="drop"
)

sp["ybin"] = pd.qcut(
    sp.latitude,
    5,
    labels=False,
    duplicates="drop"
)

rows = []

for (
    xbin,
    ybin
), group in sp.groupby(
    ["xbin", "ybin"],
    observed=True
):

    if len(group) < N_PER_CELL:
        continue

    group = group.nlargest(
        N_PER_CELL,
        "magnitude"
    )

    # use the lowest 100 magnitude-ranked events
    # after sorting magnitude
    group = group.sort_values(
        "magnitude"
    )

    q = maxc(
        group.magnitude
    )

    rows.append(
        [
            group.longitude.mean(),
            group.latitude.mean(),
            q,
            len(group)
        ]
    )

mcmap = pd.DataFrame(
    rows,
    columns=[
        "longitude",
        "latitude",
        "Mc",
        "N"
    ]
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

if len(mcmap):

    s = ax.scatter(
        mcmap.longitude,
        mcmap.latitude,
        c=mcmap.Mc,
        s=120,
        cmap="viridis"
    )

    fig.colorbar(
        s,
        ax=ax,
        label="Mc"
    )

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.grid(alpha=0.25)

savefig(
    fig,
    12,
    "Spatial magnitude of completeness"
)


# ============================================================
# FIGURE 13
# b IN SPACE + UNCERTAINTY
# ============================================================

rows = []

for (
    xbin,
    ybin
), group in sp.groupby(
    ["xbin", "ybin"],
    observed=True
):

    if len(group) < N_PER_CELL:
        continue

    group = group.sort_values(
        "magnitude"
    ).tail(N_PER_CELL)

    q = maxc(
        group.magnitude
    )

    x = group[
        group.magnitude >= q
    ].magnitude.to_numpy()

    if len(x) < 10:
        continue

    b = aki_b(
        x,
        q
    )

    sigma = (
        2.3
        * b
        * b
        * np.std(
            x,
            ddof=1
        )
        / np.sqrt(
            len(x)
            * (len(x) - 1)
        )
    )

    rows.append(
        [
            group.longitude.mean(),
            group.latitude.mean(),
            b,
            sigma
        ]
    )

bmap = pd.DataFrame(
    rows,
    columns=[
        "longitude",
        "latitude",
        "b",
        "sigma"
    ]
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

if len(bmap):

    s = ax.scatter(
        bmap.longitude,
        bmap.latitude,
        c=bmap.b,
        s=130,
        cmap="viridis"
    )

    fig.colorbar(
        s,
        ax=ax,
        label="b-value"
    )

    ax.errorbar(
        bmap.longitude,
        bmap.latitude,
        yerr=bmap.sigma * 0.05,
        fmt="none",
        alpha=0.5
    )

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.grid(alpha=0.25)

savefig(
    fig,
    13,
    "Spatial b-value with uncertainty"
)


# ============================================================
# FIGURES 14-16
# FIT REAL ETAS MODEL
# ============================================================

print("\nPreparing ETAS sequence...")

# Use the largest-event sequence rather than fitting all 9417
# events. This keeps the O(N^2) E-step computationally feasible.

seq = sc[
    (
        sc.time >= t0
    )
    &
    (
        sc.time <=
        t0 + pd.Timedelta(days=30)
    )
].copy()

# Apply completeness threshold
seq = seq[
    seq.magnitude >= mc
].copy()

# Keep manageable size
MAX_EVENTS = 800

if len(seq) > MAX_EVENTS:

    seq = seq.head(
        MAX_EVENTS
    )

print(
    "ETAS fitting sequence:",
    len(seq),
    "events"
)

# convert time to days
times = (
    (
        seq.time
        - seq.time.iloc[0]
    )
    / pd.Timedelta(days=1)
).to_numpy()

mags = (
    seq.magnitude
    .to_numpy()
)

M0 = mc

duration = (
    times[-1]
    - times[0]
)

# ------------------------------------------------------------
# IMPORT ETAS MODEL
# ------------------------------------------------------------

from eq_toolkit.calibrate.em import (
    run_em,
    run_em_restarts
)

from eq_toolkit.calibrate.mstep import (
    ETASParameters
)

from eq_toolkit.model.intensity import (
    temporal_intensity
)

# ------------------------------------------------------------
# INITIAL PARAMETER SETS
# ------------------------------------------------------------

rate = len(times) / max(
    duration,
    1e-6
)

starts = [

    ETASParameters(
        mu=max(rate * 0.3, 1e-4),
        K=0.1,
        alpha=0.5,
        c=0.01,
        p=1.2
    ),

    ETASParameters(
        mu=max(rate * 0.5, 1e-4),
        K=0.2,
        alpha=0.8,
        c=0.01,
        p=1.5
    ),

    ETASParameters(
        mu=max(rate * 0.7, 1e-4),
        K=0.3,
        alpha=1.0,
        c=0.05,
        p=1.8
    )
]

print(
    "\nRunning ETAS EM..."
)

result = run_em_restarts(
    times,
    mags,
    starts,
    m0=M0,
    max_iterations=30,
    tolerance=1e-4
)

params = result.parameters

print("\nFINAL ETAS PARAMETERS")

print(
    "mu    =",
    params.mu
)

print(
    "K     =",
    params.K
)

print(
    "alpha =",
    params.alpha
)

print(
    "c     =",
    params.c
)

print(
    "p     =",
    params.p
)

print(
    "iterations =",
    result.iterations
)

print(
    "converged =",
    result.converged
)


# ============================================================
# FIGURE 14
# OMORI + PRODUCTIVITY
# ============================================================

tau = np.logspace(
    -4,
    2,
    400
)

omori = (
    (params.p - 1)
    * params.c ** (
        params.p - 1
    )
    * (
        tau + params.c
    ) ** (-params.p)
)

magnitudes14 = np.linspace(
    M0,
    max(
        M0 + 3,
        mags.max()
    ),
    100
)

productivity = (
    params.K
    * 10 ** (
        params.alpha
        * (
            magnitudes14
            - M0
        )
    )
)

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.loglog(
    tau,
    omori,
    label="Omori decay"
)

ax.set_xlabel(
    "Time since triggering event"
)

ax.set_ylabel(
    "Normalized Omori kernel"
)

ax2 = ax.twinx()

ax2.semilogy(
    magnitudes14,
    productivity,
    "--",
    label="Productivity"
)

ax2.set_ylabel(
    "Expected productivity"
)

savefig(
    fig,
    14,
    "ETAS triggering kernel and productivity"
)


# ============================================================
# FIGURE 15
# CONDITIONAL INTENSITY
# ============================================================

lam = temporal_intensity(
    times,
    mags,
    mu=params.mu,
    K=params.K,
    alpha=params.alpha,
    M0=M0,
    c=params.c,
    p=params.p
)

background = np.full(
    len(lam),
    params.mu
)

triggered = np.maximum(
    lam - background,
    1e-12
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    times,
    lam,
    label="Total conditional intensity"
)

ax.plot(
    times,
    background,
    "--",
    label="Background"
)

ax.plot(
    times,
    triggered,
    label="Triggered component"
)

ax.set_xlabel(
    "Time since start (days)"
)

ax.set_ylabel(
    "Conditional intensity"
)

ax.legend()

ax.grid(alpha=0.25)

savefig(
    fig,
    15,
    "Conditional intensity: background and triggering"
)


# ============================================================
# FIGURE 16
# REAL VS SIMULATED CATALOG
# ============================================================

def simulate_etas(
    duration,
    mu,
    K,
    alpha,
    M0,
    c,
    p,
    nmax=1000,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    times_sim = list(
        rng.uniform(
            0,
            duration,
            rng.poisson(
                max(mu * duration, 1)
            )
        )
    )

    mags_sim = list(
        M0
        + rng.exponential(
            0.5,
            len(times_sim)
        )
    )

    i = 0

    while (
        i < len(times_sim)
        and len(times_sim) < nmax
    ):

        ti = times_sim[i]
        mi = mags_sim[i]

        productivity = (K* 10 ** (alpha* (mi - M0)) )

        nk = rng.poisson(
            max(
                productivity,
                0
            )
        )

        for _ in range(nk):

            u = rng.random()

            if p <= 1:
                continue

            delay = (c* ((1 - u)** (-1/ (p - 1))- 1))

            tj = ti + delay

            if tj < duration:

                times_sim.append(tj)

                mags_sim.append(M0+ rng.exponential(0.5))

                if (len(times_sim)>= nmax):
                    break

        i += 1

    order = np.argsort(times_sim)

    return (
        np.asarray(
            times_sim
        )[order],
        np.asarray(
            mags_sim
        )[order]
    )


sim_t, sim_m = simulate_etas(
    duration,
    params.mu,
    params.K,
    params.alpha,
    M0,
    params.c,
    params.p,
    nmax=len(seq),
    seed=42
)

fig, ax = plt.subplots(
    figsize=(11, 5)
)

ax.scatter(
    times,
    mags,
    s=12,
    alpha=0.6,
    label="Real SCSN sequence"
)

ax.scatter(
    sim_t,
    sim_m,
    s=12,
    alpha=0.6,
    label="ETAS simulated sequence"
)

ax.set_xlabel(
    "Time since sequence start (days)"
)

ax.set_ylabel(
    "Magnitude"
)

ax.legend()

ax.grid(alpha=0.25)

savefig(
    fig,
    16,
    "Real sequence versus ETAS simulated catalog"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("FIGURES 01-16 COMPLETE")
print("=" * 60)

for i in range(1, 17):

    p = OUT / f"fig{i:02d}.png"

    print(
        f"{i:02d}:",
        "OK" if p.exists() else "MISSING"
    )

print(
    "\nOutput directory:"
)

print(
    OUT.resolve()
)