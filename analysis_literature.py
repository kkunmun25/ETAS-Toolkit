import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# SETTINGS
# ============================================================

CATALOG = "sc-catalog.csv"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)


# Inspect the raw SCSN file
with open("sc-catalog.txt", "r", errors="ignore") as f:
    lines = f.readlines()

print("Total lines:", len(lines))

for i, line in enumerate(lines[:30]):
    print(i, repr(line))

# Show lines containing likely earthquake data
for i, line in enumerate(lines):
    if "</PRE>" in line or "DATE" in line.upper() or "TIME" in line.upper():
        print(i, repr(line[:300]))

import pandas as pd
import re

records = []

with open("sc-catalog.txt", "r", errors="ignore") as f:
    for line in f:
        line = line.strip()

        # Keep only earthquake data rows beginning with YYYY/MM/DD
        if re.match(r"^\d{4}/\d{2}/\d{2}", line):
            parts = line.split()

            # Corrected indices based on raw file inspection:
            # 0:Date, 1:Time, 2:ET, 3:GT, 4:MAG, 5:M, 6:LAT, 7:LON, 8:DEPTH, 9:Q, 10:EVID
            records.append({
                "time": parts[0] + " " + parts[1],
                "magnitude": float(parts[4]),
                "latitude": float(parts[6]),
                "longitude": float(parts[7]),
                "depth": float(parts[8]),
                "event_id": parts[10]
            })

df = pd.DataFrame(records)

# Convert time
df["time"] = pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M:%S.%f")

print("DF shape:", df.shape)
df.head()
df.tail()

print("\nDF columns:")
print(df.columns.tolist())            

# ============================================================
# LOAD CATALOG
# ============================================================




# ============================================================
# COMMON FMD FUNCTION
# ============================================================

def fmd_counts(magnitudes, bin_width=0.1):

    mmin = np.floor(magnitudes.min() / bin_width) * bin_width
    mmax = np.ceil(magnitudes.max() / bin_width) * bin_width

    bins = np.arange(
        mmin,
        mmax + bin_width,
        bin_width
    )

    counts, edges = np.histogram(
        magnitudes,
        bins=bins
    )

    centers = edges[:-1] + bin_width / 2

    cumulative = np.cumsum(counts[::-1])[::-1]

    return centers, counts, cumulative


# ============================================================
# SIMPLE MAXIMUM-CURVATURE Mc
# ============================================================

def max_curvature_mc(magnitudes, bin_width=0.1):

    centers, counts, cumulative = fmd_counts(
        magnitudes,
        bin_width
    )

    if len(counts) == 0:
        return np.nan

    idx = np.argmax(counts)

    return centers[idx]


# ============================================================
# GFT Mc
# ============================================================

def gft_mc(
    magnitudes,
    bin_width=0.1,
    gof_threshold=0.90
):

    centers, counts, cumulative = fmd_counts(
        magnitudes,
        bin_width
    )

    positive = cumulative > 0

    centers = centers[positive]
    cumulative = cumulative[positive]

    if len(centers) < 5:
        return np.nan

    logN = np.log10(cumulative)

    best_mc = np.nan

    # Test each possible completeness magnitude
    for i in range(len(centers) - 3):

        x = centers[i:]
        y = logN[i:]

        if len(x) < 4:
            continue

        slope, intercept = np.polyfit(x, y, 1)

        predicted = intercept + slope * x

        ss_res = np.sum((y - predicted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            continue

        r2 = 1 - ss_res / ss_tot

        if r2 >= gof_threshold:
            best_mc = centers[i]
            break

    return best_mc


# ============================================================
# ARGUMENT 1
# DIFFERENT Mc ESTIMATORS CAN DISAGREE
# ============================================================

print("\n" + "=" * 70)
print("ARGUMENT 1: Mc ESTIMATOR COMPARISON")
print("=" * 70)

magnitudes = df["magnitude"].values

mc_maxc = max_curvature_mc(magnitudes)
mc_gft = gft_mc(magnitudes)

# A conservative alternative based on the first stable
# Gutenberg-Richter linear portion.
# This provides a third independently calculated threshold.

centers, counts, cumulative = fmd_counts(magnitudes)

positive = cumulative > 0

x = centers[positive]
y = cumulative[positive]

logN = np.log10(y)

# Estimate a stability-based Mc by testing linearity
mc_stable = np.nan

for i in range(len(x) - 5):

    xx = x[i:]
    yy = logN[i:]

    if len(xx) < 6:
        continue

    slope, intercept = np.polyfit(xx, yy, 1)

    pred = slope * xx + intercept

    r2 = 1 - (
        np.sum((yy - pred) ** 2)
        /
        np.sum((yy - np.mean(yy)) ** 2)
    )

    if r2 >= 0.98:
        mc_stable = xx[0]
        break


print("MAXC Mc :", mc_maxc)
print("GFT Mc  :", mc_gft)
print("Stable Mc:", mc_stable)

mc_values = [
    mc_maxc,
    mc_gft,
    mc_stable
]

valid_mc = [
    x for x in mc_values
    if np.isfinite(x)
]

if valid_mc:

    mc_min = min(valid_mc)
    mc_max = max(valid_mc)

    print("Mc range:", mc_min, "-", mc_max)
    print("Maximum disagreement:", mc_max - mc_min)


# ============================================================
# FIGURE 1 — FMD + Mc ESTIMATES
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    centers,
    np.log10(np.maximum(cumulative, 1)),
    marker="o",
    markersize=3,
    linewidth=1.5,
    label="USGS catalog"
)

if np.isfinite(mc_maxc):
    plt.axvline(
        mc_maxc,
        linestyle="--",
        linewidth=2,
        label=f"MAXC Mc = {mc_maxc:.2f}"
    )

if np.isfinite(mc_gft):
    plt.axvline(
        mc_gft,
        linestyle=":",
        linewidth=2,
        label=f"GFT Mc = {mc_gft:.2f}"
    )

if np.isfinite(mc_stable):
    plt.axvline(
        mc_stable,
        linestyle="-.",
        linewidth=2,
        label=f"Stable Mc = {mc_stable:.2f}"
    )

plt.xlabel("Magnitude")
plt.ylabel(r"$\log_{10} N(\geq M)$")
plt.title("Frequency–Magnitude Distribution and Mc Estimates")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "mc_comparison.png"),
    dpi=300
)

plt.close()


# ============================================================
# IDENTIFY M7.2 MAINSHOCK
# ============================================================

mainshock = df.loc[df["magnitude"].idxmax()]

t0 = mainshock["time"]

print("\n" + "=" * 70)
print("MAINSHOCK")
print("=" * 70)

print("Time:", t0)
print("Magnitude:", mainshock["magnitude"])
print("Latitude:", mainshock["latitude"])
print("Longitude:", mainshock["longitude"])
print("Depth:", mainshock["depth"])


# ============================================================
# EXTRACT 60-DAY SEQUENCE
# ============================================================

sequence = df[
    (df["time"] > t0) &
    (df["time"] <= t0 + pd.Timedelta(days=60))
].copy()

sequence["hours"] = (
    sequence["time"] - t0
).dt.total_seconds() / 3600

print("60-day sequence events:", len(sequence))


# ============================================================
# ARGUMENT 2
# SHORT-TERM AFTERSHOCK INCOMPLETENESS
# ============================================================

print("\n" + "=" * 70)
print("ARGUMENT 2: TIME-DEPENDENT Mc")
print("=" * 70)

windows = [
    ("0–1 h", 0, 1),
    ("1–3 h", 1, 3),
    ("3–6 h", 3, 6),
    ("6–12 h", 6, 12),
    ("12–24 h", 12, 24),
    ("1–2 d", 24, 48),
    ("2–4 d", 48, 96),
    ("4–7 d", 96, 168),
    ("7–14 d", 168, 336),
    ("14–30 d", 336, 720),
    ("30–60 d", 720, 1440)
]

results = []

for label, start, end in windows:

    subset = sequence[
        (sequence["hours"] >= start) &
        (sequence["hours"] < end)
    ]

    if len(subset) >= 20:

        mc = max_curvature_mc(
            subset["magnitude"].values
        )

    else:

        mc = np.nan

    results.append(
        {
            "window": label,
            "start": start,
            "end": end,
            "events": len(subset),
            "mc": mc
        }
    )

mc_time = pd.DataFrame(results)

print(mc_time.to_string(index=False))


# ============================================================
# FIGURE 2A — Mc VS TIME
# ============================================================

mid_time = (
    mc_time["start"] +
    mc_time["end"]
) / 2

plt.figure(figsize=(10, 6))

plt.plot(
    mid_time,
    mc_time["mc"],
    marker="o",
    linewidth=2
)

plt.xlabel("Time since M7.2 mainshock (hours)")
plt.ylabel("Magnitude of completeness, Mc")
plt.title("Time Dependence of Magnitude of Completeness")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "mc_time_dependence.png"),
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 2B — MAGNITUDE VS TIME
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    sequence["hours"],
    sequence["magnitude"],
    s=8,
    alpha=0.5
)

plt.xscale("log")

plt.xlabel("Time since M7.2 mainshock (hours)")
plt.ylabel("Magnitude")
plt.title("Magnitude–Time Distribution of the Aftershock Sequence")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "magnitude_time_sequence.png"),
    dpi=300
)

plt.close()


# ============================================================
# ARGUMENT 3
# OMORI–UTSU AFTERSHOCK DECAY
# ============================================================

print("\n" + "=" * 70)
print("ARGUMENT 3: AFTERSHOCK TEMPORAL DECAY")
print("=" * 70)

# Use events above the overall catalog completeness threshold
# to reduce incompleteness bias.

if valid_mc:
    completeness = max(valid_mc)
else:
    completeness = 2.7

clean_sequence = sequence[
    sequence["magnitude"] >= completeness
].copy()

# logarithmic time bins
time_edges = np.logspace(
    np.log10(0.01),
    np.log10(1440),
    30
)

counts, edges = np.histogram(
    clean_sequence["hours"],
    bins=time_edges
)

centers_time = np.sqrt(
    edges[:-1] * edges[1:]
)

duration = np.diff(edges)

rate = counts / duration

valid = (
    (counts > 0) &
    np.isfinite(rate)
)

t = centers_time[valid]
r = rate[valid]


# Omori-Utsu model
def omori_utsu(t, K, c, p):

    return K / ((t + c) ** p)


# Initial parameters
p0 = [
    max(r),
    0.1,
    1.0
]

bounds = (
    [0, 0, 0.1],
    [np.inf, np.inf, 3]
)

try:

    popt, pcov = curve_fit(
        omori_utsu,
        t,
        r,
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )

    K, c, p = popt

    print(f"K = {K:.4f}")
    print(f"c = {c:.4f} hours")
    print(f"p = {p:.4f}")

except Exception as e:

    print("Omori-Utsu fit failed:", e)

    K = np.nan
    c = np.nan
    p = np.nan


# ============================================================
# FIGURE 3 — OMORI-UTSU
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    t,
    r,
    s=35,
    label="Observed aftershock rate"
)

if np.isfinite(p):

    tt = np.logspace(
        np.log10(t.min()),
        np.log10(t.max()),
        300
    )

    plt.plot(
        tt,
        omori_utsu(tt, K, c, p),
        linewidth=2,
        label=f"Omori–Utsu fit: p = {p:.2f}"
    )

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Time since mainshock (hours)")
plt.ylabel("Aftershock rate (events/hour)")
plt.title("Temporal Decay of the M7.2 Aftershock Sequence")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "aftershock_decay.png"),
    dpi=300
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nFigures created:")

for f in os.listdir(OUTDIR):
    print("  ", os.path.join(OUTDIR, f))

print("\nKey results:")

if valid_mc:
    print(
        f"Mc disagreement = "
        f"{mc_max - mc_min:.2f} magnitude units"
    )

print(
    f"Mainshock = M{mainshock['magnitude']:.1f}"
)

print(
    f"Aftershocks in 60 days = {len(sequence)}"
)

if np.isfinite(p):
    print(
        f"Omori-Utsu p = {p:.3f}"
    )