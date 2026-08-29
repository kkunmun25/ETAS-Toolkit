from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd

from eq_toolkit.catalog.model import Catalog

from eq_toolkit.sources.comcat import get_events as comcat_get_events
from eq_toolkit.sources.isc import get_events as isc_get_events
from eq_toolkit.sources.fdsn import FDSNClient
from eq_toolkit.sources.geo_net import GeoNet
from eq_toolkit.sources.afad import AFAD

from eq_toolkit.sources.scrape.auth_greece import AUTHGreece
from eq_toolkit.sources.scrape.csn_chile import CSN_Chile


# ============================================================
# OUTPUT FOLDERS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "phase2_catalogs"
NORMALIZED_DIR = DATA_DIR / "normalized"
METADATA_DIR = DATA_DIR / "metadata"
CACHE_DIR = DATA_DIR / "cache"

for folder in [
    NORMALIZED_DIR,
    METADATA_DIR,
    CACHE_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# SMALL SMOKE-TEST PERIOD
# ============================================================

START_TIME = "2024-01-01"
END_TIME = "2024-01-02"

MIN_MAG = 3.0


# ============================================================
# REGIONS
# ============================================================

# Normal FDSN bbox order:
# west, south, east, north

CALIFORNIA = (
    -125.0,
    32.0,
    -113.0,
    42.5,
)

TURKEY = (
    25.0,
    35.0,
    45.0,
    43.0,
)

NEW_ZEALAND = (
    170.0,
    -48.0,
    180.0,
    -34.0,
)

GREECE = (
    19.0,
    34.0,
    29.0,
    42.0,
)

CHILE = (
    -76.0,
    -56.0,
    -66.0,
    -17.0,
)


# ============================================================
# SAVE HELPERS
# ============================================================

def save_catalog(catalog, source_name):

    filename = (
        NORMALIZED_DIR
        / f"{source_name.lower()}_catalog.csv"
    )

    catalog.save_csv(filename)

    return filename


def save_metadata(
    source,
    source_type,
    region,
    event_count,
    status,
    output=None,
    error=None,
):

    metadata = {
        "source": source,
        "source_type": source_type,
        "region": region,
        "start_time": START_TIME,
        "end_time": END_TIME,
        "minimum_magnitude": MIN_MAG,
        "event_count": event_count,
        "status": status,
        "output_file": str(output) if output else None,
        "error": error,
        "download_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    filename = (
        METADATA_DIR
        / f"{source.lower()}_metadata.json"
    )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def print_success(source, catalog):

    print(
        f"\n[SUCCESS] {source}: "
        f"{len(catalog.data)} events"
    )


def print_failed(source, exc):

    print(f"\n[FAILED] {source}")
    print(f"Reason: {exc}")


# ============================================================
# 1. USGS COMCAT
# ============================================================

def download_usgs():

    source = "USGS_ComCat"

    print("\n" + "=" * 60)
    print("1/8 USGS COMCAT")
    print("=" * 60)

    try:

        west, south, east, north = CALIFORNIA

        catalog = comcat_get_events(
            starttime=START_TIME,
            endtime=END_TIME,
            minmagnitude=MIN_MAG,
            minlatitude=south,
            maxlatitude=north,
            minlongitude=west,
            maxlongitude=east,
        )

        output = save_catalog(
            catalog,
            source,
        )

        save_metadata(
            source,
            "API",
            "California",
            len(catalog.data),
            "success",
            output,
        )

        print_success(
            source,
            catalog,
        )

        return True

    except Exception as exc:

        print_failed(
            source,
            exc,
        )

        save_metadata(
            source,
            "API",
            "California",
            0,
            "failed",
            error=str(exc),
        )

        return False


# ============================================================
# 2. ISC
# ============================================================

def download_isc():

    source = "ISC"

    print("\n" + "=" * 60)
    print("2/8 ISC")
    print("=" * 60)

    try:

        west, south, east, north = CALIFORNIA

        catalog = isc_get_events(
            starttime=START_TIME,
            endtime=END_TIME,
            minmagnitude=MIN_MAG,
            minlatitude=south,
            maxlatitude=north,
            minlongitude=west,
            maxlongitude=east,
        )

        output = save_catalog(
            catalog,
            source,
        )

        save_metadata(
            source,
            "API",
            "California",
            len(catalog.data),
            "success",
            output,
        )

        print_success(
            source,
            catalog,
        )

        return True

    except Exception as exc:

        print_failed(
            source,
            exc,
        )

        save_metadata(
            source,
            "API",
            "California",
            0,
            "failed",
            error=str(exc),
        )

        return False


# ============================================================
# 3. SCEDC
# ============================================================

def download_scedc():

    source = "SCEDC"

    print("\n" + "=" * 60)
    print("3/8 SCEDC")
    print("=" * 60)

    try:

        client = FDSNClient.from_server(
            "SCEDC"
        )

        catalog = client.get_events(
            bbox=CALIFORNIA,
            time_range=(
                START_TIME,
                END_TIME,
            ),
            min_mag=MIN_MAG,
        )

        output = save_catalog(
            catalog,
            source,
        )

        save_metadata(
            source,
            "FDSN API",
            "California",
            len(catalog.data),
            "success",
            output,
        )

        print_success(
            source,
            catalog,
        )

        return True

    except Exception as exc:

        print_failed(
            source,
            exc,
        )

        save_metadata(
            source,
            "FDSN API",
            "California",
            0,
            "failed",
            error=str(exc),
        )

        return False


# ============================================================
# 4. NCEDC
# ============================================================

def download_ncedc():

    source = "NCEDC"

    print("\n" + "=" * 60)
    print("4/8 NCEDC")
    print("=" * 60)

    try:

        client = FDSNClient.from_server(
            "NCEDC"
        )

        catalog = client.get_events(
            bbox=CALIFORNIA,
            time_range=(
                START_TIME,
                END_TIME,
            ),
            min_mag=MIN_MAG,
        )

        output = save_catalog(
            catalog,
            source,
        )

        save_metadata(
            source,
            "FDSN API",
            "California",
            len(catalog.data),
            "success",
            output,
        )

        print_success(
            source,
            catalog,
        )

        return True

    except Exception as exc:

        print_failed(
            source,
            exc,
        )

        save_metadata(
            source,
            "FDSN API",
            "California",
            0,
            "failed",
            error=str(exc),
        )

        return False


# ============================================================
# 5. AFAD TURKEY
# ============================================================

def download_afad():

    source = "AFAD"

    print("\n" + "=" * 60)
    print("AFAD TURKEY")
    print("=" * 60)

    try:

        client = AFAD()

        catalog = client.get_events(
            starttime="2024-01-01",
            endtime="2024-01-10",
            minlatitude=35,
            maxlatitude=43,
            minlongitude=25,
            maxlongitude=45,
            minmagnitude=4.0,
        )

        output = save_catalog(
            catalog,
            source,
        )

        save_metadata(
            source,
            "REST API",
            "Turkey",
            len(catalog.data),
            "success",
            output,
        )

        print_success(
            source,
            catalog,
        )

        return True

    except Exception as exc:

        print_failed(source, exc)

        return False
    


# ============================================================
# 6. GEONET NEW ZEALAND
# ============================================================

def download_geonet():

    source = "GeoNet"

    print("\n" + "=" * 60)
    print("6/8 GEONET NEW ZEALAND")
    print("=" * 60)

    try:

        client = GeoNet()

        # IMPORTANT:
        # GeoNet code uses:
        # min_lon, max_lon, min_lat, max_lat

        geonet_bbox = (
            170.0,
            180.0,
            -48.0,
            -34.0,
        )

        catalog = client.get_events(
            bbox=geonet_bbox,
            time_range=(
                START_TIME,
                END_TIME,
            ),
            min_mag=MIN_MAG,
        )

        output = save_catalog(
            catalog,
            source,
        )

        save_metadata(
            source,
            "WFS API",
            "New Zealand",
            len(catalog.data),
            "success",
            output,
        )

        print_success(
            source,
            catalog,
        )

        return True

    except Exception as exc:

        print_failed(
            source,
            exc,
        )

        save_metadata(
            source,
            "WFS API",
            "New Zealand",
            0,
            "failed",
            error=str(exc),
        )

        return False


# ============================================================
# 7. AUTH GREECE SCRAPER
# ============================================================

def download_auth():

    source = "AUTH_Greece"

    print("\n" + "=" * 60)
    print("AUTH GREECE - SCRAPED")
    print("=" * 60)

    try:

        scraper = AUTHGreece()

        catalog = scraper.get_events(
            bbox=(
                34.0,
                42.0,
                19.0,
                29.0,
            ),
            min_mag=5.0,
        )

        output = save_catalog(
            catalog,
            source,
        )

        save_metadata(
            source,
            "Scraped/Bulk",
            "Greece",
            len(catalog.data),
            "success",
            output,
        )

        print_success(
            source,
            catalog,
        )

        return True

    except Exception as exc:

        print_failed(
            source,
            exc,
        )

        return False

     

# ============================================================
# 8. CSN CHILE SCRAPER
# ============================================================

def download_csn():

    source = "CSN_Chile"

    print("\n" + "=" * 60)
    print("8/8 CSN CHILE - SCRAPED")
    print("=" * 60)

    try:

        scraper = CSN_Chile()

        catalog = scraper.get_catalog()

        df = catalog.data.copy()

        # ----------------------------------------
        # Convert columns safely
        # ----------------------------------------

        df["time"] = pd.to_datetime(
            df["time"],
            errors="coerce",
        )

        df["magnitude"] = pd.to_numeric(
            df["magnitude"],
            errors="coerce",
        )

        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce",
        )

        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce",
        )

        # ----------------------------------------
        # Apply filters locally
        # ----------------------------------------

        start = pd.to_datetime(
            START_TIME
        )

        end = pd.to_datetime(
            END_TIME
        )

        west, south, east, north = CHILE

        df = df[
            (df["magnitude"] >= MIN_MAG)
        ]

        df = df[
            (df["latitude"] >= south)
            & (df["latitude"] <= north)
            & (df["longitude"] >= west)
            & (df["longitude"] <= east)
        ]

        # CSN webpage may contain only recent events.
        # Only apply date filtering when valid dates exist.

        valid_time = df["time"].notna()

        if valid_time.any():

            date_filtered = df[
                valid_time
                & (df["time"] >= start)
                & (df["time"] < end)
            ]

            if len(date_filtered) > 0:
                df = date_filtered

        df = df.reset_index(drop=True)

        catalog.data = df

        output = save_catalog(
            catalog,
            source,
        )

        save_metadata(
            source,
            "Scraper",
            "Chile",
            len(catalog.data),
            "success",
            output,
        )

        print_success(
            source,
            catalog,
        )

        return True

    except Exception as exc:

        print_failed(
            source,
            exc,
        )

        save_metadata(
            source,
            "Scraper",
            "Chile",
            0,
            "failed",
            error=str(exc),
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ETAS TOOLKIT - PHASE 2 CATALOG DOWNLOAD")
    print("=" * 70)

    print(f"API test period : {START_TIME} to {END_TIME}")
    print(f"Minimum Mag     : {MIN_MAG}")
    print()

    results = {}

    results["USGS_ComCat"] = download_usgs()
    results["ISC"] = download_isc()
    results["NCEDC"] = download_ncedc()
    results["AFAD"] = download_afad()
    results["GeoNet"] = download_geonet()
    results["AUTH_Greece"] = download_auth()
    results["CSN_Chile"] = download_csn()

    print()
    print("=" * 70)
    print("FINAL DOWNLOAD SUMMARY")
    print("=" * 70)

    successful = 0

    for source, status in results.items():

        if status:
            print(f"[SUCCESS] {source}")
            successful += 1
        else:
            print(f"[FAILED ] {source}")

    print()
    print(
        f"Successful sources: "
        f"{successful}/{len(results)}"
    )

    print()
    print("Saved catalogs:")

    for filename in sorted(
        NORMALIZED_DIR.glob("*.csv")
    ):

        try:

            df = pd.read_csv(
                filename
            )

            print(
                f"  {filename.name:<35}"
                f"{len(df):>8} events"
            )

        except Exception:

            print(
                f"  {filename.name}"
            )

    print()
    print(
        f"Catalog folder: {NORMALIZED_DIR}"
    )

    print(
        f"Metadata folder: {METADATA_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()