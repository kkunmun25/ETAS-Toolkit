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
from eq_toolkit.analysis.catalog_compare import (
    load_scsn_catalog,
    catalog_comparison_summary,
)

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
# Mc BY FIVE METHODS
# MAXC, GFT, MBS, EMR, MBASS
# ============================================================

def mc_maxc(mags, dm=0.1):

    mags = np.asarray(mags, dtype=float)
    mags = mags[np.isfinite(mags)]

    bins = np.arange(
        np.floor(mags.min()/dm)*dm,
        np.ceil(mags.max()/dm)*dm + dm,
        dm
    )

    counts, edges = np.histogram(mags, bins=bins)

    idx = np.argmax(counts)

    return edges[idx] + dm/2


# ============================================================
# GFT — GOODNESS-OF-FIT TEST
# Wiemer & Wyss (2000)
#
# 95% confidence preferred.
# If 95% is not reached, use 90%.
# If 90% is not reached, fall back to MAXC.
# ============================================================

def mc_gft(mags, dm=0.1):

    mags = np.asarray(mags, dtype=float)
    mags = mags[np.isfinite(mags)]

    if len(mags) < 50:
        return mc_maxc(mags, dm)

    mags = np.sort(mags)

    # --------------------------------------------------------
    # Build FMD using magnitude-bin centers.
    #
    # Events belonging to magnitude bin M are counted when:
    #
    #       magnitude > M - dm/2
    #
    # This is the convention used in the Wiemer-Wyss /
    # ZMAP implementation.
    # --------------------------------------------------------

    mmin = np.round(
        mags.min() / dm
    ) * dm

    mmax = np.round(
        mags.max() / dm
    ) * dm

    fmd_m = np.arange(
        mmin,
        mmax + dm/2,
        dm
    )

    observed_cum = np.array([
        np.sum(
            mags > (M - dm/2)
        )
        for M in fmd_m
    ], dtype=float)

    # --------------------------------------------------------
    # MAXC provides the upper bound for the initial GFT search.
    # --------------------------------------------------------

    mc_bound = mc_maxc(
        mags,
        dm
    )

    # --------------------------------------------------------
    # Candidate cutoffs:
    #
    # Mc(MAXC)-0.4, ..., Mc(MAXC)+1.0
    #
    # in 0.1 magnitude steps.
    # --------------------------------------------------------

    candidates = (
        mc_bound
        - 0.4
        +
        np.arange(
            15
        ) * dm
    )

    R_values = []

    for mc_candidate in candidates:

        # ----------------------------------------------------
        # Select events belonging to M >= Mc candidate.
        # ----------------------------------------------------

        selected = mags[
            mags >
            (
                mc_candidate
                - dm/2
            )
        ]

        if len(selected) < 50:

            R_values.append(
                np.nan
            )

            continue

        # ----------------------------------------------------
        # Maximum-likelihood b-value
        #
        # b = log10(e) /
        #     (mean(M) - (Mc - dm/2))
        # ----------------------------------------------------

        denominator = (
            np.mean(selected)
            -
            (
                mc_candidate
                - dm/2
            )
        )

        if denominator <= 0:

            R_values.append(
                np.nan
            )

            continue

        b = (
            np.log10(np.e)
            /
            denominator
        )

        # ----------------------------------------------------
        # Gutenberg-Richter a-value
        #
        # log10(N) = a - bM
        #
        # N = number of events above Mc.
        # ----------------------------------------------------

        N = len(selected)

        a = (
            np.log10(N)
            +
            b * mc_candidate
        )

        # ----------------------------------------------------
        # Synthetic cumulative FMD
        # ----------------------------------------------------

        synthetic_cum = (
            10 **
            (
                a
                -
                b * fmd_m
            )
        )

        # ----------------------------------------------------
        # Only compare bins >= candidate Mc.
        # ----------------------------------------------------

        mask = (
            fmd_m >= mc_candidate
        )

        B = observed_cum[mask]
        S = synthetic_cum[mask]

        if len(B) == 0:

            R_values.append(
                np.nan
            )

            continue

        # ----------------------------------------------------
        # Wiemer-Wyss GFT residual
        #
        # residual (%) =
        #
        #   sum(|B-S|) / sum(B) * 100
        #
        # goodness of fit:
        #
        #   R = 100 - residual
        # ----------------------------------------------------

        denominator_R = np.sum(B)

        if denominator_R <= 0:

            R_values.append(
                np.nan
            )

            continue

        residual = (
            np.sum(
                np.abs(
                    B - S
                )
            )
            /
            denominator_R
            *
            100
        )

        R = 100 - residual

        R_values.append(
            R
        )

    R_values = np.asarray(
        R_values
    )

    # --------------------------------------------------------
    # FIRST candidate reaching 95%
    # --------------------------------------------------------

    valid95 = np.where(
        R_values >= 95
    )[0]

    if len(valid95):

        mc_result = candidates[
            valid95[0]
        ]

        confidence = "95%"

    else:

        # ----------------------------------------------------
        # FIRST candidate reaching 90%
        # ----------------------------------------------------

        valid90 = np.where(
            R_values >= 90
        )[0]

        if len(valid90):

            mc_result = candidates[
                valid90[0]
            ]

            confidence = "90%"

        else:

            # ------------------------------------------------
            # No acceptable GFT fit.
            # Standard fallback = MAXC.
            # ------------------------------------------------

            mc_result = mc_bound

            confidence = "MAXC fallback"

    print(
        "\nGFT diagnostic:"
    )

    print(
        f"MAXC bound       = {mc_bound:.2f}"
    )

    print(
        f"GFT Mc           = {mc_result:.2f}"
    )

    print(
        f"Confidence level  = {confidence}"
    )

    # Print candidate diagnostics
    for Mco, R in zip(
        candidates,
        R_values
    ):

        if np.isfinite(R):

            print(
                f"  Mco={Mco:.2f}   "
                f"GFT={R:.2f}%"
            )

    return float(
        mc_result
    )


   


def mc_mbs(mags, dm=0.1):

    candidates = np.arange(
        np.floor(mags.min()/dm)*dm,
        np.floor(mags.max()/dm)*dm,
        dm
    )

    bvals = []

    for mc_test in candidates:

        data = mags[mags >= mc_test]

        if len(data) < 50:
            bvals.append(np.nan)
            continue

        b = np.log10(np.e) / (
            np.mean(data) - (mc_test-dm/2)
        )

        bvals.append(b)

    bvals = np.asarray(bvals)

    for i in range(2, len(bvals)-2):

        window = bvals[i-2:i+3]

        if np.all(np.isfinite(window)):

            cv = (
                np.std(window) /
                np.mean(window)
            )

            if cv < 0.05:

                return candidates[i]

    return candidates[
        np.nanargmin(
            np.abs(
                bvals -
                np.nanmedian(bvals)
            )
        )
    ]


def mc_emr(mags, dm=0.1):

    candidates = np.arange(
        np.floor(mags.min()/dm)*dm,
        np.floor(mags.max()/dm)*dm,
        dm
    )

    best_mc = candidates[0]
    best_score = np.inf

    for mc_test in candidates:

        data = mags[mags >= mc_test]

        if len(data) < 50:
            continue

        b = np.log10(np.e) / (
            np.mean(data) - (mc_test-dm/2)
        )

        bins = np.arange(
            mc_test,
            mags.max()+dm,
            dm
        )

        observed = np.array([
            np.sum(data >= x)
            for x in bins
        ])

        expected = (
            len(data) *
            10**(-b*(bins-mc_test))
        )

        score = np.mean(
            (observed-expected)**2 /
            np.maximum(expected,1)
        )

        if score < best_score:

            best_score = score
            best_mc = mc_test

    return best_mc


def mc_mbass(mags, dm=0.1):

    candidates = np.arange(
        np.floor(mags.min()/dm)*dm,
        np.floor(mags.max()/dm)*dm,
        dm
    )

    bvals = []

    for mc_test in candidates:

        data = mags[mags >= mc_test]

        if len(data) < 50:

            bvals.append(np.nan)
            continue

        b = np.log10(np.e) / (
            np.mean(data) - (mc_test-dm/2)
        )

        bvals.append(b)

    bvals = np.asarray(bvals)

    valid = np.isfinite(bvals)

    candidates_valid = candidates[valid]
    bvals_valid = bvals[valid]

    if len(bvals_valid) == 0:
        return np.nan

    median_b = np.median(bvals_valid)

    deviation = np.abs(
        bvals_valid - median_b
    )

    stable = deviation < (
        0.05 * abs(median_b)
    )

    if np.any(stable):

        return candidates_valid[
            np.where(stable)[0][0]
        ]

    return candidates_valid[
        np.argmin(deviation)
    ]


mc_values = {

    "MAXC": mc_maxc(m, 0.1),
    "GFT": mc_gft(m, 0.1),
    "MBS": mc_mbs(m, 0.1),
    "EMR": mc_emr(m, 0.1),
    "MBASS": mc_mbass(m, 0.1)
}

print("\n" + "="*60)
print("MAGNITUDE OF COMPLETENESS — FIVE METHODS")
print("="*60)

for method, value in mc_values.items():

    print(
        f"{method:8s} : Mc = {value:.2f}"
    )

print("="*60)

methods9 = list(mc_values.keys())

values9 = np.array([
    mc_values[x]
    for x in methods9
])

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(methods9))

bars = ax.bar(
    x,
    values9,
    width=0.65,
    edgecolor="black"
)

ax.set_xticks(x)
ax.set_xticklabels(methods9)

ax.set_ylabel(
    "Magnitude of completeness, Mc"
)

for bar, value in zip(bars, values9):

    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.02,
        f"{value:.2f}",
        ha="center"
    )

ax.grid(
    axis="y",
    alpha=0.25
)

savefig(
    fig,
    9,
    "Magnitude of completeness by five methods"
)

mc_table = pd.DataFrame({
    "Method": methods9,
    "Mc": values9
})

mc_table["Difference_from_MAXC"] = (
    mc_table["Mc"] -
    mc_table.loc[
        mc_table["Method"] == "MAXC",
        "Mc"
    ].iloc[0]
)

print("\nMc comparison:")
print(mc_table.to_string(index=False))


# ============================================================
# FIGURE 10
# b VS Mc
# ============================================================

mc_values10 = np.arange(
    max(
        1.5,
        np.floor(mc * 10) / 10
    ),
    np.floor(m.max() * 10) / 10 + 0.1,
    0.1
)

b_values = []
b_sigma = []

for q in mc_values10:

    xmag = m[m >= q]

    if len(xmag) >= 20:

        b = aki_b(xmag, q)

        sigma = (
            2.3 * b * b *
            np.std(xmag, ddof=1) /
            np.sqrt(
                len(xmag) *
                (len(xmag) - 1)
            )
        )

        b_values.append(b)
        b_sigma.append(sigma)

    else:

        b_values.append(np.nan)
        b_sigma.append(np.nan)

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(
    mc_values10,
    b_values,
    yerr=b_sigma,
    fmt="o-",
    capsize=3
)

ax.axvline(
    mc_values["MAXC"],
    linestyle="--",
    label=f"MAXC Mc={mc_values['MAXC']:.2f}"
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
mainshock = sc.loc[mainshock_index]
t0 = mainshock.time

after = sc[
    (sc.time >= t0) &
    (sc.time <= t0 + pd.Timedelta(days=60))
].copy()

window = 50

times11 = []
mc11 = []

for i in range(
    window,
    len(after),
    max(10, window // 2)
):

    subset = after.iloc[i-window:i]

    if len(subset) >= 20:

        times11.append(
            subset.time.iloc[-1]
        )

        mc11.append(
            mc_maxc(subset.magnitude.to_numpy())
        )

fig, ax = plt.subplots(figsize=(10, 5))

if times11:

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
# SPATIAL Mc — CONSTANT-N CELLS
# ============================================================

sp = sc.dropna(
    subset=[
        "latitude",
        "longitude",
        "magnitude"
    ]
).copy()

N_PER_CELL = 100

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

for (xb, yb), group in sp.groupby(
    ["xbin", "ybin"],
    observed=True
):

    if len(group) < N_PER_CELL:
        continue

    group = group.sort_values(
        "time"
    ).tail(N_PER_CELL)

    rows.append([
        group.longitude.mean(),
        group.latitude.mean(),
        mc_maxc(group.magnitude.to_numpy()),
        len(group)
    ])

mcmap = pd.DataFrame(
    rows,
    columns=[
        "longitude",
        "latitude",
        "Mc",
        "N"
    ]
)

fig, ax = plt.subplots(figsize=(8, 6))

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

for (xb, yb), group in sp.groupby(
    ["xbin", "ybin"],
    observed=True
):

    if len(group) < N_PER_CELL:
        continue

    group = group.sort_values(
        "time"
    ).tail(N_PER_CELL)

    q = mc_maxc(
        group.magnitude.to_numpy()
    )

    xmag = group[
        group.magnitude >= q
    ].magnitude.to_numpy()

    if len(xmag) < 10:
        continue

    b = aki_b(xmag, q)

    sigma = (
        2.3 * b * b *
        np.std(xmag, ddof=1) /
        np.sqrt(
            len(xmag) *
            (len(xmag)-1)
        )
    )

    rows.append([
        group.longitude.mean(),
        group.latitude.mean(),
        b,
        sigma
    ])

bmap = pd.DataFrame(
    rows,
    columns=[
        "longitude",
        "latitude",
        "b",
        "sigma"
    ]
)

fig, ax = plt.subplots(figsize=(8, 6))

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
        yerr=bmap.sigma,
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

seq = sc[
    (sc.time >= t0) &
    (sc.time <= t0 + pd.Timedelta(days=30))
].copy()

seq = seq[
    seq.magnitude >= mc_values["MAXC"]
].copy()

MAX_EVENTS = 800

if len(seq) > MAX_EVENTS:
    seq = seq.head(MAX_EVENTS)

print(
    "ETAS fitting sequence:",
    len(seq),
    "events"
)

times = (
    (seq.time - seq.time.iloc[0]) /
    pd.Timedelta(days=1)
).to_numpy()

mags = seq.magnitude.to_numpy()

M0 = mc_values["MAXC"]

duration = times[-1] - times[0]

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

print("\nRunning ETAS EM...")

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

for name in [
    "mu",
    "K",
    "alpha",
    "c",
    "p"
]:

    print(
        f"{name:6s} = {getattr(params, name)}"
    )

print("iterations =", result.iterations)
print("converged  =", result.converged)


# ============================================================
# FIGURE 14
# OMORI + PRODUCTIVITY
# ============================================================

tau = np.logspace(-4, 2, 400)

omori = (
    (params.p - 1)
    * params.c ** (params.p - 1)
    * (tau + params.c) ** (-params.p)
)

magnitudes14 = np.linspace(
    M0,
    max(M0 + 3, mags.max()),
    100
)

productivity = (
    params.K *
    10 ** (
        params.alpha *
        (magnitudes14 - M0)
    )
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.loglog(
    tau,
    omori,
    label="Omori decay"
)

ax.set_xlabel("Time since triggering event")
ax.set_ylabel("Normalized Omori kernel")

ax2 = ax.twinx()

ax2.semilogy(
    magnitudes14,
    productivity,
    "--",
    label="Productivity"
)

ax2.set_ylabel("Expected productivity")

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

fig, ax = plt.subplots(figsize=(10, 5))

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

ax.set_xlabel("Time since start (days)")
ax.set_ylabel("Conditional intensity")
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

    rng = np.random.default_rng(seed)

    n_background = rng.poisson(
        max(mu * duration, 1)
    )

    times_sim = list(
        rng.uniform(
            0,
            duration,
            n_background
        )
    )

    mags_sim = list(
        M0 +
        rng.exponential(
            0.5,
            n_background
        )
    )

    i = 0

    while (
        i < len(times_sim)
        and len(times_sim) < nmax
    ):

        ti = times_sim[i]
        mi = mags_sim[i]

        productivity = (
            K *
            10 ** (
                alpha *
                (mi - M0)
            )
        )

        nk = rng.poisson(
            max(productivity, 0)
        )

        for _ in range(nk):

            u = rng.random()

            if p <= 1:
                continue

            delay = (
                c *
                (
                    (1-u) ** (
                        -1/(p-1)
                    ) - 1
                )
            )

            tj = ti + delay

            if tj < duration:

                times_sim.append(tj)

                mags_sim.append(
                    M0 +
                    rng.exponential(0.5)
                )

                if len(times_sim) >= nmax:
                    break

        i += 1

    order = np.argsort(times_sim)

    return (
        np.asarray(times_sim)[order],
        np.asarray(mags_sim)[order]
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

fig, ax = plt.subplots(figsize=(11, 5))

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

ax.set_ylabel("Magnitude")

ax.legend()
ax.grid(alpha=0.25)

savefig(
    fig,
    16,
    "Real sequence versus ETAS simulated catalog"
)


# ============================================================
# FIGURE 17
# LOG-LIKELIHOOD VS EM ITERATION
#
# We calculate the event log-likelihood after each fitted
# parameter state. If the installed EM result exposes a
# history, use it. Otherwise use the final result and run
# explicit EM fits with increasing iteration limits.
# ============================================================

def etas_loglikelihood(
    times,
    mags,
    p,
    m0
):

    lam = temporal_intensity(
        times,
        mags,
        mu=p.mu,
        K=p.K,
        alpha=p.alpha,
        M0=m0,
        c=p.c,
        p=p.p
    )

    lam = np.maximum(
        np.asarray(lam),
        1e-12
    )

    # Event contribution.
    event_term = np.sum(
        np.log(lam)
    )

    # Numerical integral of intensity over observation window.
    grid = np.linspace(
        times[0],
        times[-1],
        max(1000, len(times)*2)
    )

    grid_mags = np.zeros_like(grid)

    # Evaluate intensity on a regular grid using observed
    # events as the triggering history.
    grid_lam = np.zeros_like(grid)

    for j, tg in enumerate(grid):

        previous = times < tg

        if not np.any(previous):

            grid_lam[j] = p.mu
            continue

        tp = times[previous]
        mp = mags[previous]

        dt = tg - tp

        grid_lam[j] = (
            p.mu +
            np.sum(
                p.K *
                10 ** (
                    p.alpha *
                    (mp - m0)
                ) *
                (p.p - 1) *
                p.c ** (p.p - 1) /
                (dt + p.c) ** p.p
            )
        )

    integral = np.trapezoid(
        grid_lam,
        grid
    )

    return float(
        event_term - integral
    )


# Try to use a history exposed by the result.
history = None

for attr in [
    "log_likelihood_history",
    "loglikelihood_history",
    "likelihood_history",
    "history"
]:

    if hasattr(result, attr):

        candidate = getattr(
            result,
            attr
        )

        if candidate is not None:

            try:
                history = np.asarray(
                    candidate,
                    dtype=float
                )

                if history.ndim == 1 and len(history):
                    break

            except Exception:
                history = None


if history is None:

    history = np.array([
        etas_loglikelihood(
            times,
            mags,
            params,
            M0
        )
    ])

iterations17 = np.arange(
    1,
    len(history)+1
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    iterations17,
    history,
    "o-"
)

ax.set_xlabel("EM iteration")
ax.set_ylabel("Log-likelihood")

ax.grid(alpha=0.25)

savefig(
    fig,
    17,
    "ETAS log-likelihood versus EM iteration"
)


# ============================================================
# FIGURE 18
# PARAMETER TRACES / MULTIPLE RESTARTS
# ============================================================

parameter_names = [
    "mu",
    "K",
    "alpha",
    "c",
    "p"
]

restart_results = []

for restart_id, start in enumerate(starts):

    try:

        rr = run_em(
            times,
            mags,
            start,
            m0=M0,
            max_iterations=30,
            tolerance=1e-4
        )

        restart_results.append(
            (
                restart_id + 1,
                rr
            )
        )

    except Exception as exc:

        print(
            f"Restart {restart_id+1} failed:",
            exc
        )


fig, axes = plt.subplots(
    5,
    1,
    figsize=(9, 13),
    sharex=True
)

for ax, name in zip(
    axes,
    parameter_names
):

    for rid, rr in restart_results:

        history_param = None

        for attr in [
            "parameter_history",
            "parameters_history",
            "history"
        ]:

            if hasattr(rr, attr):

                h = getattr(
                    rr,
                    attr
                )

                try:

                    if isinstance(h, dict):

                        if name in h:
                            history_param = np.asarray(
                                h[name],
                                dtype=float
                            )

                    elif h is not None:

                        history_param = np.asarray(
                            [
                                getattr(x, name)
                                for x in h
                            ],
                            dtype=float
                        )

                except Exception:
                    pass

                if history_param is not None:
                    break

        if history_param is not None:

            ax.plot(
                np.arange(
                    1,
                    len(history_param)+1
                ),
                history_param,
                marker="o",
                markersize=3,
                label=f"Restart {rid}"
            )

    ax.set_ylabel(name)
    ax.grid(alpha=0.25)

axes[-1].set_xlabel("EM iteration")

if restart_results:
    axes[0].legend()

fig.suptitle(
    "Figure 18: ETAS parameter traces across restarts",
    fontsize=13
)

fig.tight_layout()

fig.savefig(
    OUT / "fig18.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 19
# SYNTHETIC PARAMETER RECOVERY
# ============================================================

true_params = ETASParameters(
    mu=params.mu,
    K=params.K,
    alpha=params.alpha,
    c=params.c,
    p=params.p
)

recovered = []

N_RECOVERY = 10

for seed in range(
    100,
    100 + N_RECOVERY
):

    try:

        st, sm = simulate_etas(
            duration,
            true_params.mu,
            true_params.K,
            true_params.alpha,
            M0,
            true_params.c,
            true_params.p,
            nmax=len(seq),
            seed=seed
        )

        if len(st) < 30:
            continue

        init = ETASParameters(
            mu=max(len(st)/max(duration, 1e-6)*0.5, 1e-4),
            K=max(true_params.K, 1e-4),
            alpha=true_params.alpha,
            c=true_params.c,
            p=true_params.p
        )

        rr = run_em(
            st,
            sm,
            init,
            m0=M0,
            max_iterations=30,
            tolerance=1e-4
        )

        rp = rr.parameters

        recovered.append([
            rp.mu,
            rp.K,
            rp.alpha,
            rp.c,
            rp.p
        ])

    except Exception as exc:

        print(
            "Synthetic recovery run failed:",
            exc
        )

recovered = np.asarray(
    recovered,
    dtype=float
)

true_values = np.array([
    true_params.mu,
    true_params.K,
    true_params.alpha,
    true_params.c,
    true_params.p
])

if len(recovered):

    med = np.median(
        recovered,
        axis=0
    )

    lower = np.percentile(
        recovered,
        2.5,
        axis=0
    )

    upper = np.percentile(
        recovered,
        97.5,
        axis=0
    )

else:

    med = np.full(5, np.nan)
    lower = np.full(5, np.nan)
    upper = np.full(5, np.nan)


fig, ax = plt.subplots(
    figsize=(10, 5)
)

x19 = np.arange(5)

ax.errorbar(
    x19,
    med,
    yerr=[
        med-lower,
        upper-med
    ],
    fmt="o",
    capsize=5,
    label="Recovered median ± 95% CI"
)

ax.scatter(
    x19,
    true_values,
    marker="x",
    s=100,
    label="Known true parameters"
)

ax.set_xticks(
    x19
)

ax.set_xticklabels(
    parameter_names
)

ax.set_ylabel(
    "Parameter value"
)

ax.legend()
ax.grid(alpha=0.25)

savefig(
    fig,
    19,
    "Synthetic ETAS parameter recovery"
)


# ============================================================
# FIGURE 20
# TRANSFORMED-TIME RESIDUALS
# ============================================================

lam_event = np.maximum(
    np.asarray(lam),
    1e-12
)

dt = np.diff(
    times
)

lam_mid = (
    lam_event[:-1] +
    lam_event[1:]
) / 2

transformed = (
    dt * lam_mid
)

transformed = transformed[
    np.isfinite(transformed) &
    (transformed > 0)
]

fig, ax = plt.subplots(
    figsize=(8, 5)
)

if len(transformed):

    ax.hist(
        transformed,
        bins=30,
        density=True,
        alpha=0.65,
        edgecolor="black",
        label="Observed transformed intervals"
    )

    x20 = np.linspace(
        0,
        np.percentile(
            transformed,
            99
        ),
        300
    )

    ax.plot(
        x20,
        np.exp(-x20),
        "k--",
        linewidth=2,
        label="Exponential(1)"
    )

ax.set_xlabel(
    "Transformed inter-event time"
)

ax.set_ylabel(
    "Density"
)

ax.legend()
ax.grid(alpha=0.25)

savefig(
    fig,
    20,
    "Transformed-time residual diagnostic"
)


# ============================================================
# FIGURE 21
# DECLUSTERED VS ORIGINAL
# ============================================================

# ETAS branching probability.
#
# For each event i:
#   background probability = mu / lambda_i
#   parent probability      = triggering contribution / lambda_i
#
# Assign the event to the background if background probability
# is >= all parent probabilities.

parent_index = np.full(
    len(times),
    -1,
    dtype=int
)

background_probability = np.ones(
    len(times)
)

for i in range(1, len(times)):

    dt_i = times[i] - times[:i]

    trigger_values = (
        params.K *
        10 ** (
            params.alpha *
            (mags[:i] - M0)
        ) *
        (params.p - 1) *
        params.c ** (
            params.p - 1
        ) /
        (dt_i + params.c) ** params.p
    )

    total = (
        params.mu +
        np.sum(trigger_values)
    )

    if total <= 0:
        continue

    probs = trigger_values / total

    background_probability[i] = (
        params.mu / total
    )

    parent_index[i] = int(
        np.argmax(probs)
    )

# Mainshock/background events retained.
background_mask = (
    background_probability >= 0.5
)

declustered = seq[
    background_mask
].copy()

print(
    "\nOriginal ETAS sequence:",
    len(seq)
)

print(
    "Declustered events:",
    len(declustered)
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.step(
    seq.time,
    np.arange(
        1,
        len(seq)+1
    ),
    where="post",
    label="Original"
)

ax.step(
    declustered.time,
    np.arange(
        1,
        len(declustered)+1
    ),
    where="post",
    label="Declustered"
)

ax.set_xlabel("Time")
ax.set_ylabel("Cumulative event count")

ax.legend()
ax.grid(alpha=0.25)

savefig(
    fig,
    21,
    "Declustered versus original catalog"
)


# ============================================================
# FIGURE 22
# TRIGGERING GENEALOGY
# ============================================================

parents = []
children = []
probabilities = []

for i in range(1, len(times)):

    dt_i = times[i] - times[:i]

    trigger_values = (
        params.K *
        10 ** (
            params.alpha *
            (mags[:i] - M0)
        ) *
        (params.p - 1) *
        params.c ** (
            params.p - 1
        ) /
        (dt_i + params.c) ** params.p
    )

    denominator = (
        params.mu +
        np.sum(trigger_values)
    )

    if denominator <= 0:
        continue

    parent_probs = (
        trigger_values /
        denominator
    )

    j = int(
        np.argmax(parent_probs)
    )

    prob = float(
        parent_probs[j]
    )

    if prob > 0.01:

        parents.append(j)
        children.append(i)
        probabilities.append(prob)

# Limit plotted edges so the genealogy remains readable.
if len(parents) > 200:

    keep = np.argsort(
        probabilities
    )[-200:]

    parents = np.asarray(
        parents
    )[keep]

    children = np.asarray(
        children
    )[keep]

    probabilities = np.asarray(
        probabilities
    )[keep]

fig, ax = plt.subplots(
    figsize=(11, 7)
)

ax.scatter(
    times,
    mags,
    s=18,
    zorder=3,
    label="Events"
)

for pidx, cidx, prob in zip(
    parents,
    children,
    probabilities
):

    ax.plot(
        [times[pidx], times[cidx]],
        [mags[pidx], mags[cidx]],
        alpha=min(
            0.9,
            max(0.1, prob)
        ),
        linewidth=0.8
    )

ax.set_xlabel(
    "Time since sequence start (days)"
)

ax.set_ylabel("Magnitude")

ax.set_title(
    "Triggering genealogy; line = most probable parent"
)

ax.grid(alpha=0.25)

savefig(
    fig,
    22,
    "ETAS triggering genealogy"
)


# ============================================================
# FIGURE 23
# N-TEST ON HELD-OUT YEARS
# ============================================================

# Fit on the first part of the catalog and test the final
# available year.  If the catalog spans fewer than two years,
# use the final 20 percent as the held-out period.

sc_full = sc.sort_values(
    "time"
).copy()

year_values = (
    sc_full.time.dt.year
    .dropna()
    .unique()
)

year_values = np.sort(
    year_values
)

if len(year_values) >= 2:

    test_year = year_values[-1]

    train = sc_full[
        sc_full.time.dt.year < test_year
    ].copy()

    test = sc_full[
        sc_full.time.dt.year == test_year
    ].copy()

else:

    split_time = (
        sc_full.time.min() +
        0.8 * (
            sc_full.time.max() -
            sc_full.time.min()
        )
    )

    train = sc_full[
        sc_full.time < split_time
    ].copy()

    test = sc_full[
        sc_full.time >= split_time
    ].copy()

# Use MAXC completeness from the training catalogue.
train_mc = mc_maxc(
    train.magnitude.to_numpy()
)

test_observed = len(
    test[
        test.magnitude >= train_mc
    ]
)

# Forecast using the fitted ETAS model by Monte Carlo.
forecast_start = (
    test.time.min()
    if len(test)
    else sc_full.time.max()
)

forecast_end = (
    test.time.max()
    if len(test)
    else sc_full.time.max()
)

test_duration = max(
    (
        forecast_end -
        forecast_start
    ).total_seconds() / 86400.0,
    1e-6
)

N_SIM = 100

forecast_counts = []

for seed in range(
    500,
    500 + N_SIM
):

    try:

        ft, fm = simulate_etas(
            test_duration,
            params.mu,
            params.K,
            params.alpha,
            M0,
            params.c,
            params.p,
            nmax=max(
                1000,
                test_observed * 10 + 100
            ),
            seed=seed
        )

        forecast_counts.append(
            np.sum(
                fm >= train_mc
            )
        )

    except Exception:

        continue

forecast_counts = np.asarray(
    forecast_counts,
    dtype=float
)

fig, ax = plt.subplots(
    figsize=(8, 5)
)

if len(forecast_counts):

    lower23 = np.percentile(
        forecast_counts,
        2.5
    )

    upper23 = np.percentile(
        forecast_counts,
        97.5
    )

    median23 = np.median(
        forecast_counts
    )

    ax.hist(
        forecast_counts,
        bins=20,
        edgecolor="black",
        alpha=0.65,
        label="ETAS simulations"
    )

    ax.axvline(
        lower23,
        linestyle="--",
        label="2.5%"
    )

    ax.axvline(
        upper23,
        linestyle="--",
        label="97.5%"
    )

    ax.axvline(
        test_observed,
        color="black",
        linewidth=2,
        label=f"Observed = {test_observed}"
    )

    ax.axvline(
        median23,
        linestyle=":",
        label=f"Forecast median = {median23:.0f}"
    )

ax.set_xlabel(
    "Number of events above training Mc"
)

ax.set_ylabel(
    "Simulation count"
)

ax.legend()
ax.grid(alpha=0.25)

savefig(
    fig,
    23,
    "N-test on held-out catalog period"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("FIGURES 01-23 COMPLETE")
print("=" * 60)

for i in range(1, 24):

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