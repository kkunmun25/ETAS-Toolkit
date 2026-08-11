"""
eq_toolkit/sources/scrape/auth_greece.py

Download earthquake catalogue data from the
Aristotle University of Thessaloniki (AUTH), Greece.
"""

import io
from datetime import datetime

import pandas as pd
import requests

from eq_toolkit.catalog.model import Catalog


class AUTHGreece:
    """
    AUTH Greece earthquake catalogue scraper.
    """

    BASE_URL = (
        "https://seismo.auth.gr/wp-content/uploads/2023/03/seiscat.dat"
    )

    def __init__(self):
        pass

    def get_raw_data(self):
        """
        Download the raw AUTH earthquake catalogue.
        """

        response = requests.get(
            self.BASE_URL,
            timeout=30,
        )

        response.raise_for_status()

        return response.text

    def parse_line(self, line):
        """
        Parse one earthquake line from the AUTH catalogue.

        AUTH records can look like:

        ? -550 0000000000.00 36.9000 22.4000 0. 6.8

        or:

        -490 0000000000.00 37.4000 25.3000 0. 6.0
        """

        parts = line.split()

        # Ignore empty lines
        if not parts:
            return None

        # Remove optional catalogue flag
        if parts[0] in {"?", "!"}:
            parts = parts[1:]

        # A valid earthquake record has 6 fields
        if len(parts) != 6:
            return None

        # Check that the first field is a valid year
        try:
            year = int(parts[0])
        except ValueError:
            return None

        # Convert numerical values
        try:
            latitude = float(parts[2])
            longitude = float(parts[3])
            depth = float(parts[4])
            magnitude = float(parts[5])
        except ValueError:
            return None

        return {
            "year": year,
            "date_string": parts[1],
            "latitude": latitude,
            "longitude": longitude,
            "depth": depth,
            "magnitude": magnitude,
        }

    def parse_catalog(self, text):
        """
        Parse the complete AUTH text catalogue.
        """

        events = []

        for line in text.splitlines():

            event = self.parse_line(line)

            if event is not None:
                events.append(event)

        return events

    def _make_time(self, year, date_string):
        """
        Convert AUTH year + MoDaHrMnSecs into a datetime.

        AUTH uses:
        MoDaHrMnSecs

        Example:
        0101000000.00
        = January 1, 00:00:00

        BCE years cannot be represented by Python datetime,
        so they are returned as NaT.
        """

        # Python datetime cannot represent BCE years
        if year < 1:
            return pd.NaT

        try:
            value = date_string.split(".")[0]

            # Make sure the string has 10 digits
            value = value.zfill(10)

            month = int(value[0:2])
            day = int(value[2:4])
            hour = int(value[4:6])
            minute = int(value[6:8])
            second = int(value[8:10])

            return datetime(
                year,
                month if month >= 1 else 1,
                day if day >= 1 else 1,
                hour,
                minute,
                second,
            )

        except (ValueError, TypeError):
            return pd.NaT

    def get_events(
        self,
        bbox=None,
        time_range=None,
        min_mag=None,
    ):
        """
        Download and return AUTH earthquakes as a Catalog.

        Parameters
        ----------
        bbox : tuple, optional
            (min_lat, max_lat, min_lon, max_lon)

        time_range : tuple, optional
            (start_time, end_time)

        min_mag : float, optional
            Minimum earthquake magnitude.

        Returns
        -------
        Catalog
            Earthquake catalogue.
        """

        text = self.get_raw_data()

        events = self.parse_catalog(text)

        rows = []

        for i, event in enumerate(events):

            event_time = self._make_time(
                event["year"],
                event["date_string"],
            )

            rows.append(
                {
                    "time": event_time,
                    "latitude": event["latitude"],
                    "longitude": event["longitude"],
                    "depth": event["depth"],
                    "magnitude": event["magnitude"],
                    "magnitude_type": "M",
                    "source_agency": "AUTH",
                    "event_id": f"AUTH-{i + 1}",
                }
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "time",
                "latitude",
                "longitude",
                "depth",
                "magnitude",
                "magnitude_type",
                "source_agency",
                "event_id",
            ],
        )

    

        if bbox is not None:

            min_lat, max_lat, min_lon, max_lon = bbox

            df = df[
                (df["latitude"] >= min_lat)
                & (df["latitude"] <= max_lat)
                & (df["longitude"] >= min_lon)
                & (df["longitude"] <= max_lon)
            ]


        if min_mag is not None:

            df = df[df["magnitude"] >= min_mag]
 
        if time_range is not None:

            start_time, end_time = time_range

            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)

            df = df[
                (df["time"] >= start_time)
                & (df["time"] <= end_time)
            ]

  

        df = df.reset_index(drop=True)

        catalog = Catalog()

        for _, row in df.iterrows():
         catalog.add_event(
        time=row["time"],
        latitude=row["latitude"],
        longitude=row["longitude"],
        depth=row["depth"],
        magnitude=row["magnitude"],
        magnitude_type=row["magnitude_type"],
        source_agency=row["source_agency"],
        event_id=row["event_id"],
    )

        return catalog