"""
Download and parse Global CMT catalogs.
"""

from pathlib import Path

from obspy import read_events

from eq_toolkit.catalog.model import Catalog   
import requests

GCMT_URL = ("https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec20.ndk")

def download_ndk(outfile: str | Path) -> Path:
    """
    Download the Global CMT NDK catalog.
    """
    outfile = Path(outfile)

    response = requests.get(GCMT_URL, timeout=30)
    response.raise_for_status()

    outfile.write_bytes(response.content)
    return outfile


def get_events(ndk_file):
    """
    Read a GCMT NDK file and return a Catalog.
    """
    events = read_events(ndk_file, format="NDK")

    catalog = Catalog()

    for event in events:
        origin = event.preferred_origin() or event.origins[0]
        magnitude = event.preferred_magnitude() or event.magnitudes[0]

        catalog.add_event(
            time=origin.time.datetime,
            latitude=origin.latitude,
            longitude=origin.longitude,
            depth=origin.depth / 1000.0,
            magnitude=magnitude.mag,
            magnitude_type=magnitude.magnitude_type,
            source_agency="GCMT",
            event_id=event.resource_id.id,
        )

    return catalog