"""
Catalog comparison utilities.

This module provides functions for comparing two earthquake catalogs
covering the same region and time period.

Main comparisons:
1. Event counts
2. Common-event matching
3. Magnitude differences
4. Epicentral location differences
5. Magnitude-dependent catalog differences
6. Temporal reporting-rate differences
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "time",
    "latitude",
    "longitude",
    "magnitude",
}


def _validate_catalog(catalog: pd.DataFrame, name: str = "catalog") -> None:
    """Validate that a catalog contains the required columns."""
    missing = REQUIRED_COLUMNS - set(catalog.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing)}"
        )


def _prepare_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned copy of an earthquake catalog.

    Ensures:
    - time is datetime
    - latitude, longitude and magnitude are numeric
    - invalid rows are removed
    """
    _validate_catalog(catalog)

    df = catalog.copy()

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

    for column in ["latitude", "longitude", "magnitude"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["time", "latitude", "longitude", "magnitude"]
    ).copy()

    return df.sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------------
# 1. Event count comparison
# ---------------------------------------------------------------------

def compare_event_counts(
    catalog_a: pd.DataFrame,
    catalog_b: pd.DataFrame,
) -> dict:
    """
    Compare basic event counts between two catalogs.

    Returns
    -------
    dict
        Dictionary containing counts and time ranges.
    """
    a = _prepare_catalog(catalog_a)
    b = _prepare_catalog(catalog_b)

    return {
        "catalog_a_count": len(a),
        "catalog_b_count": len(b),
        "catalog_a_start": a["time"].min(),
        "catalog_a_end": a["time"].max(),
        "catalog_b_start": b["time"].min(),
        "catalog_b_end": b["time"].max(),
        "count_difference": len(a) - len(b),
        "count_ratio_a_to_b": (
            len(a) / len(b) if len(b) > 0 else np.nan
        ),
    }


# ---------------------------------------------------------------------
# 2. Magnitude-dependent count comparison
# ---------------------------------------------------------------------

def magnitude_count_comparison(
    catalog_a: pd.DataFrame,
    catalog_b: pd.DataFrame,
    magnitude_bins: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compare event counts by magnitude bin.

    This directly addresses the question:

        "Where does the difference sit in magnitude?"

    Parameters
    ----------
    catalog_a, catalog_b
        Earthquake catalogs.

    magnitude_bins
        Magnitude bin edges. If None, bins from the combined catalogs
        are generated at 0.1 magnitude intervals.

    Returns
    -------
    pandas.DataFrame
        Counts for both catalogs and their difference.
    """
    a = _prepare_catalog(catalog_a)
    b = _prepare_catalog(catalog_b)

    if magnitude_bins is None:
        min_mag = np.floor(
            min(a["magnitude"].min(), b["magnitude"].min()) * 10
        ) / 10

        max_mag = np.ceil(
            max(a["magnitude"].max(), b["magnitude"].max()) * 10
        ) / 10

        magnitude_bins = np.arange(
            min_mag,
            max_mag + 0.1,
            0.1,
        )

    count_a, edges = np.histogram(
        a["magnitude"],
        bins=magnitude_bins,
    )

    count_b, _ = np.histogram(
        b["magnitude"],
        bins=magnitude_bins,
    )

    centers = (edges[:-1] + edges[1:]) / 2

    result = pd.DataFrame(
        {
            "magnitude": centers,
            "count_a": count_a,
            "count_b": count_b,
        }
    )

    result["difference"] = (
        result["count_a"] - result["count_b"]
    )

    result["ratio_a_to_b"] = np.where(
        result["count_b"] > 0,
        result["count_a"] / result["count_b"],
        np.nan,
    )

    return result


# ---------------------------------------------------------------------
# 3. Common-event matching
# ---------------------------------------------------------------------

def match_common_events(
    catalog_a: pd.DataFrame,
    catalog_b: pd.DataFrame,
    time_tolerance_seconds: float = 10.0,
    distance_tolerance_km: float = 20.0,
) -> pd.DataFrame:
    """
    Match earthquakes appearing in both catalogs.

    Matching is based on:
    - origin time
    - epicentral distance

    Parameters
    ----------
    time_tolerance_seconds
        Maximum allowed difference in origin time.

    distance_tolerance_km
        Maximum allowed epicentral separation.

    Returns
    -------
    pandas.DataFrame
        One row per matched earthquake.

    Notes
    -----
    This is a practical catalog-matching method, not a replacement
    for authoritative event IDs when those are available.
    """
    a = _prepare_catalog(catalog_a)
    b = _prepare_catalog(catalog_b)

    a = a.reset_index().rename(columns={"index": "event_a"})
    b = b.reset_index().rename(columns={"index": "event_b"})

    matches = []

    # Convert time to seconds since epoch for efficient comparison.
    time_a = a["time"].astype("int64") / 1e9
    time_b = b["time"].astype("int64") / 1e9

    for i, row_a in a.iterrows():

        dt = np.abs(time_b - time_a.iloc[i])

        candidates = np.where(
            dt <= time_tolerance_seconds
        )[0]

        if len(candidates) == 0:
            continue

        # Calculate approximate epicentral distance.
        lat1 = np.radians(row_a["latitude"])
        lon1 = np.radians(row_a["longitude"])

        lat2 = np.radians(
            b.iloc[candidates]["latitude"].to_numpy()
        )
        lon2 = np.radians(
            b.iloc[candidates]["longitude"].to_numpy()
        )

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        haversine = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2) ** 2
        )

        distance = (
            2
            * 6371.0
            * np.arcsin(np.sqrt(haversine))
        )

        valid = np.where(
            distance <= distance_tolerance_km
        )[0]

        if len(valid) == 0:
            continue

        # Select closest event in space/time combination.
        best_local = valid[
            np.argmin(distance[valid])
        ]

        j = candidates[best_local]

        row_b = b.iloc[j]

        matches.append(
            {
                "event_a": row_a["event_a"],
                "event_b": row_b["event_b"],
                "time_a": row_a["time"],
                "time_b": row_b["time"],
                "time_difference_seconds": (
                    row_b["time"] - row_a["time"]
                ).total_seconds(),
                "latitude_a": row_a["latitude"],
                "longitude_a": row_a["longitude"],
                "latitude_b": row_b["latitude"],
                "longitude_b": row_b["longitude"],
                "magnitude_a": row_a["magnitude"],
                "magnitude_b": row_b["magnitude"],
                "magnitude_difference": (
                    row_a["magnitude"]
                    - row_b["magnitude"]
                ),
                "distance_km": float(
                    distance[best_local]
                ),
            }
        )

    return pd.DataFrame(matches)


# ---------------------------------------------------------------------
# 4. Magnitude comparison
# ---------------------------------------------------------------------

def compare_magnitudes(
    matched_events: pd.DataFrame,
) -> dict:
    """
    Calculate statistics for magnitudes of common events.
    """
    if matched_events.empty:
        return {
            "n_common_events": 0,
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "std_difference": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
        }

    diff = matched_events[
        "magnitude_difference"
    ].to_numpy()

    return {
        "n_common_events": len(diff),
        "mean_difference": float(np.mean(diff)),
        "median_difference": float(np.median(diff)),
        "std_difference": float(np.std(diff, ddof=1))
        if len(diff) > 1
        else 0.0,
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
    }


# ---------------------------------------------------------------------
# 5. Location comparison
# ---------------------------------------------------------------------

def compare_locations(
    matched_events: pd.DataFrame,
) -> dict:
    """
    Calculate statistics for epicentral separation.
    """
    if matched_events.empty:
        return {
            "n_common_events": 0,
            "mean_distance_km": np.nan,
            "median_distance_km": np.nan,
            "std_distance_km": np.nan,
            "max_distance_km": np.nan,
        }

    distance = matched_events[
        "distance_km"
    ].to_numpy()

    return {
        "n_common_events": len(distance),
        "mean_distance_km": float(np.mean(distance)),
        "median_distance_km": float(np.median(distance)),
        "std_distance_km": float(
            np.std(distance, ddof=1)
        )
        if len(distance) > 1
        else 0.0,
        "max_distance_km": float(np.max(distance)),
    }


# ---------------------------------------------------------------------
# 6. Temporal reporting-rate comparison
# ---------------------------------------------------------------------

def reporting_rate_comparison(
    catalog_a: pd.DataFrame,
    catalog_b: pd.DataFrame,
    frequency: str = "D",
) -> pd.DataFrame:
    """
    Compare temporal earthquake reporting rates.

    Parameters
    ----------
    frequency
        Pandas resampling frequency.

        Examples:
        "D"  -> daily
        "W"  -> weekly
        "ME" -> monthly

    Returns
    -------
    pandas.DataFrame
        Time-indexed counts for both catalogs.
    """
    a = _prepare_catalog(catalog_a)
    b = _prepare_catalog(catalog_b)

    rate_a = (
        a.set_index("time")
        .resample(frequency)
        .size()
        .rename("count_a")
    )

    rate_b = (
        b.set_index("time")
        .resample(frequency)
        .size()
        .rename("count_b")
    )

    result = pd.concat(
        [rate_a, rate_b],
        axis=1,
    ).fillna(0)

    result["difference"] = (
        result["count_a"] - result["count_b"]
    )

    return result


# ---------------------------------------------------------------------
# 7. Full comparison
# ---------------------------------------------------------------------

def compare_catalogs(
    catalog_a: pd.DataFrame,
    catalog_b: pd.DataFrame,
    time_tolerance_seconds: float = 10.0,
    distance_tolerance_km: float = 20.0,
) -> dict:
    """
    Perform the main catalog comparison workflow.

    Returns
    -------
    dict
        Results from:
        - event counts
        - magnitude-dependent counts
        - common-event matching
        - magnitude comparison
        - location comparison
    """
    counts = compare_event_counts(
        catalog_a,
        catalog_b,
    )

    magnitude_counts = magnitude_count_comparison(
        catalog_a,
        catalog_b,
    )

    matched = match_common_events(
        catalog_a,
        catalog_b,
        time_tolerance_seconds=time_tolerance_seconds,
        distance_tolerance_km=distance_tolerance_km,
    )

    magnitude_stats = compare_magnitudes(
        matched
    )

    location_stats = compare_locations(
        matched
    )

    return {
        "counts": counts,
        "magnitude_counts": magnitude_counts,
        "matched_events": matched,
        "magnitude_comparison": magnitude_stats,
        "location_comparison": location_stats,
    }

def load_scsn_catalog(filepath: str) -> pd.DataFrame:
    """
    Load an SCEDC/SCSN text catalog.

    Expected format:
    #YYY/MM/DD HH:mm:SS.ss ET GT MAG M LAT LON DEPTH Q EVID NPH NGRM
    """

    rows = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip headers and metadata
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Actual SCEDC earthquake rows contain 13 fields
            if len(parts) < 13:
                continue

            try:
                date = parts[0]
                time = parts[1]

                magnitude = float(parts[4])
                magnitude_type = parts[5]

                latitude = float(parts[6])
                longitude = float(parts[7])
                depth = float(parts[8])

                event_id = str(parts[10])

                timestamp = pd.to_datetime(
                    f"{date} {time}",
                    format="%Y/%m/%d %H:%M:%S.%f",
                    errors="coerce",
                    utc=True,
                )

                if pd.isna(timestamp):
                    continue

                rows.append(
                    {
                        "time": timestamp,
                        "magnitude": magnitude,
                        "magnitude_type": magnitude_type,
                        "latitude": latitude,
                        "longitude": longitude,
                        "depth": depth,
                        "event_id": event_id,
                    }
                )

            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No earthquake events could be parsed from {filepath}"
        )

    return (
        df.dropna(
            subset=[
                "time",
                "magnitude",
                "latitude",
                "longitude",
            ]
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

def catalog_comparison_summary(
    catalog_a: pd.DataFrame,
    catalog_b: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize catalog counts by magnitude bin.

    catalog_a = SCSN
    catalog_b = USGS
    """

    a = _prepare_catalog(catalog_a)
    b = _prepare_catalog(catalog_b)

    bins = np.arange(2.5, 7.21, 0.1)

    a_counts, edges = np.histogram(
        a["magnitude"], bins=bins
    )
    b_counts, _ = np.histogram(
        b["magnitude"], bins=bins
    )

    centers = (edges[:-1] + edges[1:]) / 2

    result = pd.DataFrame({
        "magnitude": centers,
        "scsn_count": a_counts,
        "usgs_count": b_counts,
    })

    result["difference_usgs_minus_scsn"] = (
        result["usgs_count"] - result["scsn_count"]
    )

    return result