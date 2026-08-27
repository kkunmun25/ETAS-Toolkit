from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from eq_toolkit.quality.mc import (
    maxc,
    gft,
    mbs,
    emr,
    mbass,
    b_value,
    b_value_sigma,
    lilliefors_exponentiality,
)


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "sc-catalog.txt"

if not CATALOG.exists():
    raise FileNotFoundError(f"Missing catalogue: {CATALOG}")


# ============================================================
# READ SC CATALOGUE
# ============================================================

def read_sc_catalog(path):
    magnitudes = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if not line[0].isdigit():
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            try:
                mag = float(parts[4])
            except ValueError:
                continue

            if np.isfinite(mag):
                magnitudes.append(mag)

    return np.asarray(magnitudes, dtype=float)


mags = read_sc_catalog(CATALOG)

print("\n" + "=" * 60)
print("REAL SC CATALOGUE")
print("=" * 60)
print(f"Number of events : {len(mags)}")
print(f"Minimum magnitude: {mags.min():.2f}")
print(f"Maximum magnitude: {mags.max():.2f}")
print("=" * 60)


# ============================================================
# RUN FIVE METHODS
# ============================================================

results = {}

try:
    results["MAXC"] = maxc(mags, bin_width=0.1)
except Exception as e:
    results["MAXC"] = np.nan
    print("MAXC failed:", e)

try:
    results["GFT"] = gft(
        mags,
        bin_width=0.1,
        min_fit=0.95,
        fallback_fit=0.90,
    )
except Exception as e:
    results["GFT"] = np.nan
    print("GFT failed:", e)

try:
    results["MBS"] = mbs(
        mags,
        bin_width=0.1,
        window=0.5,
    )
except Exception as e:
    results["MBS"] = np.nan
    print("MBS failed:", e)

try:
    results["EMR"] = emr(
        mags,
        bin_width=0.1,
    )
except Exception as e:
    results["EMR"] = np.nan
    print("EMR failed:", e)

try:
    results["MBASS"] = mbass(
        mags,
        bin_width=0.1,
    )
except Exception as e:
    results["MBASS"] = np.nan
    print("MBASS failed:", e)


# ============================================================
# IDENTIFY SUSPICIOUS BOUNDARY VALUES
# ============================================================

catalog_max = mags.max()
catalog_min = mags.min()

valid_results = {}
flagged_results = {}

for method, mc in results.items():

    if not np.isfinite(mc):
        flagged_results[method] = "FAILED"
        continue

    # An estimate exactly at the catalogue maximum is suspicious.
    if abs(mc - catalog_max) < 1e-9:
        flagged_results[method] = (
            f"{mc:.2f} (BOUNDARY / INVALID)"
        )
    else:
        valid_results[method] = mc


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("Mc RESULTS")
print("=" * 60)

for method, mc in results.items():

    if method in flagged_results:
        print(f"{method:8s}: {flagged_results[method]}")
    elif np.isfinite(mc):
        print(f"{method:8s}: {mc:.2f}")
    else:
        print(f"{method:8s}: FAILED")


# ============================================================
# SPREAD OF VALID METHODS
# ============================================================

valid_mc = np.array(list(valid_results.values()))

if len(valid_mc) >= 2:

    mc_min = valid_mc.min()
    mc_max = valid_mc.max()
    spread = mc_max - mc_min

    print("\n" + "=" * 60)
    print("DOCUMENTED Mc SPREAD")
    print("=" * 60)
    print(f"Lowest valid Mc : {mc_min:.2f}")
    print(f"Highest valid Mc: {mc_max:.2f}")
    print(f"Spread          : {spread:.2f}")

else:
    spread = np.nan


# ============================================================
# FREQUENCY-MAGNITUDE DISTRIBUTION
# ============================================================

bin_width = 0.1

bins = np.arange(
    np.floor(mags.min() / bin_width) * bin_width,
    np.ceil(mags.max() / bin_width) * bin_width
    + bin_width,
    bin_width,
)

counts, edges = np.histogram(mags, bins=bins)

centers = (edges[:-1] + edges[1:]) / 2

cumulative = np.array([
    np.sum(mags >= m)
    for m in centers
])


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

ax.bar(
    centers,
    counts,
    width=0.09,
    alpha=0.35,
    label="Incremental frequency",
)

ax2 = ax.twinx()

ax2.semilogy(
    centers,
    cumulative,
    "o-",
    markersize=3,
    linewidth=1.2,
    label="Cumulative frequency",
)


# ------------------------------------------------------------
# Mark Mc values
# ------------------------------------------------------------

for method, mc in results.items():

    if not np.isfinite(mc):
        continue

    if method in flagged_results:

        # EMR boundary result: mark differently
        ax.axvline(
            mc,
            linestyle=":",
            linewidth=2,
            label=f"{method} = {mc:.2f} (boundary)",
        )

    else:

        ax.axvline(
            mc,
            linestyle="--",
            linewidth=1.8,
            label=f"{method} Mc = {mc:.2f}",
        )


ax.set_xlabel("Magnitude")
ax.set_ylabel("Number of events per 0.1 magnitude")
ax2.set_ylabel("Cumulative number of events ≥ M")

ax.set_title(
    "Southern California Catalogue: "
    "Frequency–Magnitude Distribution and Mc Estimates"
)

ax.grid(alpha=0.25)

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax.legend(
    handles1 + handles2,
    labels1 + labels2,
    fontsize=9,
    loc="upper right",
)

fig.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

output = FIG_DIR / "sc_catalog_fmd_mc_methods.png"

fig.savefig(
    output,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\nFigure saved:")
print(output)



# ============================================================
# SELECTED Mc FOR DOWNSTREAM ANALYSIS
# ============================================================

selected_mc = 2.70

print("\n" + "=" * 60)
print("SELECTED Mc FOR DOWNSTREAM ANALYSIS")
print("=" * 60)
print(f"Fixed Mc = {selected_mc:.2f}")

selected = mags[mags >= selected_mc]

print(
    f"Events with M >= {selected_mc:.2f}: "
    f"{len(selected)}"
)

if len(selected) >= 20:

    b = b_value(
        selected,
        mc=selected_mc,
        bin_width=0.1,
    )

    sigma = b_value_sigma(
        selected,
        mc=selected_mc,
        bin_width=0.1,
    )

    print(f"b-value : {b:.3f}")
    print(f"sigma b : {sigma:.3f}")

# ============================================================
# LILLIEFORS / KS EXPONENTIALITY DIAGNOSTIC
# ============================================================

ks_stat, ks_p = lilliefors_exponentiality(
    mags,
    mc=selected_mc,
)

print("\n" + "=" * 60)
print("EXPONENTIALITY TEST ABOVE Mc")
print("=" * 60)

print(f"Mc              : {selected_mc:.2f}")
print(f"KS statistic    : {ks_stat:.4f}")
print(f"KS p-value      : {ks_p:.4f}")

if ks_p >= 0.05:
    print(
        "Result          : "
        "Do not reject exponentiality (p >= 0.05)"
    )
else:
    print(
        "Result          : "
        "Reject exponentiality (p < 0.05)"
    )    