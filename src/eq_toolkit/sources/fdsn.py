"""
FDSN Event Web Service Client

This module downloads earthquake catalogs from any
FDSN-compatible earthquake data center.
"""

FDSN_SERVERS = {
    "USGS": "https://earthquake.usgs.gov/fdsnws/event/1",
    "IRIS": "https://service.iris.edu/fdsnws/event/1",
    "ISC": "https://isc-mirror.iris.washington.edu/fdsnws/event/1",
    "EMSC": "https://www.seismicportal.eu/fdsnws/event/1",
    "GEOFON": "https://geofon.gfz-potsdam.de/fdsnws/event/1",
    "SCEDC": "https://service.scedc.caltech.edu/fdsnws/event/1",
    "NCEDC": "https://service.ncedc.org/fdsnws/event/1",
    "INGV": "https://webservices.ingv.it/fdsnws/event/1",
    "NOA": "https://eida.gein.noa.gr/fdsnws/event/1",
    "GeoNet": "https://service.geonet.org.nz/fdsnws/event/1",
}


import requests
from datetime import datetime
from eq_toolkit.catalog.model import Catalog


class FDSNClient:
    """
    Generic client for any FDSN Event Web Service.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")


    @classmethod
    def from_server(cls, server_name: str):
        """
        Create an FDSN client using a predefined server name.
        """
        return cls(FDSN_SERVERS[server_name])

    def _download(self, params):
        response = requests.get(
            f"{self.base_url}/query",
            params=params,
            timeout=60,
        )

        if response.status_code == 400:
            raise RuntimeError("USGS event limit exceeded")

        response.raise_for_status()

        return response.json()

    def _midpoint(self, start, end):
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)

        return (start_dt + (end_dt - start_dt) / 2).strftime("%Y-%m-%d")



    def get_events(
        self,
        bbox,
        time_range,
        min_mag
    ):
        """
        Download earthquakes from an FDSN server.
        """

        west, south, east, north = bbox
        start, end = time_range

        params = {
            "format": "geojson",
            "starttime": start,
            "endtime": end,
            "minmagnitude": min_mag,
            "minlatitude": south,
            "maxlatitude": north,
            "minlongitude": west,
            "maxlongitude": east,
        }

        try:
            data = self._download(params)
            return Catalog.from_geojson(data)

        except RuntimeError:

         mid = self._midpoint(start, end)

         first = self.get_events(
              bbox,
              (start, mid),
               min_mag,
         )

        second = self.get_events(
              bbox,
              (mid, end),
               min_mag,
    )

        first.merge(second)
        return first
     