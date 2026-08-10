"""
GeoNet earthquake catalogue downloader.

Downloads earthquake events from the GeoNet
QuakeSearch Web Feature Service (WFS) in CSV format.
"""

import io

import pandas as pd
import requests

from eq_toolkit.catalog.model import Catalog


class GeoNet:
    """
    GeoNet earthquake catalogue downloader.
    """

    BASE_URL = "https://wfs.geonet.org.nz/geonet/ows"

    def __init__(self):
        pass

    def get_events(self, bbox=None, time_range=None, min_mag=None):
        """
        Download earthquake events from GeoNet WFS.

        Parameters
        ----------
        bbox : tuple, optional
            (min_lon, max_lon, min_lat, max_lat)

        time_range : tuple, optional
            (start_time, end_time)

        min_mag : float, optional
            Minimum earthquake magnitude.

        Returns
        -------
        Catalog
            Earthquake catalogue.
        """

        filters = []

        # Magnitude filter
        if min_mag is not None:
            filters.append(f"magnitude>={min_mag}")

        # Time filter
        if time_range is not None:
            start_time, end_time = time_range

            filters.append(
                f"origintime>='{start_time}'"
            )

            filters.append(
                f"origintime<'{end_time}'"
            )

        # Bounding box filter
        if bbox is not None:
            min_lon, max_lon, min_lat, max_lat = bbox

            filters.append(
                f"BBOX(origin_geom,"
                f"{min_lon},{min_lat},"
                f"{max_lon},{max_lat})"
            )

        # Combine filters
        cql_filter = " AND ".join(filters)

        # GeoNet WFS request
        params = {
            "service": "WFS",
            "version": "1.0.0",
            "request": "GetFeature",
            "typeName": "geonet:quake_search_v1",
            "outputFormat": "csv",
        }

        if cql_filter:
            params["cql_filter"] = cql_filter

        # Download data
        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=60,
        )

        response.raise_for_status()

        # Read CSV
        df = pd.read_csv(
            io.StringIO(response.text)
        )

        # Convert GeoNet columns to Catalog columns
        catalog_df = pd.DataFrame({
            "time": pd.to_datetime(df["origintime"]),
            "latitude": pd.to_numeric(df["latitude"]),
            "longitude": pd.to_numeric(df["longitude"]),
            "depth": pd.to_numeric(df["depth"]),
            "magnitude": pd.to_numeric(df["magnitude"]),
            "magnitude_type": df["magnitudetype"],
            "source_agency": "GeoNet",
            "event_id": df["publicid"],
        })

        # Create empty Catalog
        catalog = Catalog()

        # Store the downloaded data
        catalog.data = catalog_df

        return catalog