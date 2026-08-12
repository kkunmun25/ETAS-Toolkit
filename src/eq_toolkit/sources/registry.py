"""
Source registry and command-line interface.

This module provides a common entry point for downloading
earthquake catalogs from registered earthquake sources.
"""

import argparse

from .comcat import get_events as comcat_get_events


# Region definitions

REGIONS = {
    "california": {
        "bbox": (-125.0, 32.0, -113.0, 42.5),
        "source": "comcat",
    },
}

# Source registry

SOURCES = {
    "comcat": comcat_get_events,
}


def list_regions():
    """Return all registered regions."""
    return sorted(REGIONS.keys())


def list_sources():
    """Return all registered sources."""
    return sorted(SOURCES.keys())

# Fetch function


def fetch_region(region, start_year, end_year, min_mag):
    """
    Fetch earthquake events for a named region.

    """

    region = region.lower()

    # Check whether the requested region exists
    if region not in REGIONS:
        available = ", ".join(REGIONS.keys())

        raise ValueError(
            f"Unknown region '{region}'. "
            f"Available regions: {available}"
        )

    # Get region information
    region_info = REGIONS[region]

    bbox = region_info["bbox"]
    source_name = region_info["source"]

    # Get the source function
    source_function = SOURCES[source_name]

    # Convert years to dates
    starttime = f"{start_year}-01-01"
    endtime = f"{end_year}-12-31"

   

    minlongitude = bbox[0]
    minlatitude = bbox[1]
    maxlongitude = bbox[2]
    maxlatitude = bbox[3]

    print(f"Region      : {region}")
    print(f"Source      : {source_name}")
    print(f"Start       : {starttime}")
    print(f"End         : {endtime}")
    print(f"Minimum Mag : {min_mag}")
    print(f"BBox        : {bbox}")
    print()

    # Call the registered source
    catalog = source_function(
        starttime=starttime,
        endtime=endtime,
        minmagnitude=min_mag,
        minlatitude=minlatitude,
        maxlatitude=maxlatitude,
        minlongitude=minlongitude,
        maxlongitude=maxlongitude,
    )

    return catalog


# Command-line interface


def main():
    """Run the command-line interface."""

    parser = argparse.ArgumentParser(
        description="Download earthquake catalogs from registered sources."
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )
  
    # fetch command
  

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch an earthquake catalog.",
    )

    fetch_parser.add_argument(
        "--region",
        required=True,
        help="Region name, for example: california",
    )

    fetch_parser.add_argument(
        "--from",
        dest="start_year",
        required=True,
        type=int,
        help="Starting year.",
    )

    fetch_parser.add_argument(
        "--to",
        dest="end_year",
        required=True,
        type=int,
        help="Ending year.",
    )

    fetch_parser.add_argument(
        "--min-mag",
        dest="min_mag",
        required=True,
        type=float,
        help="Minimum earthquake magnitude.",
    )

    args = parser.parse_args()

    # Execute fetch command
  

    if args.command == "fetch":

        catalog = fetch_region(
            region=args.region,
            start_year=args.start_year,
            end_year=args.end_year,
            min_mag=args.min_mag,
        )

        print()
        print("Catalog successfully downloaded.")

        try:
            print(f"Number of events: {len(catalog)}")
        except TypeError:
            print(catalog)


if __name__ == "__main__":
    main()