"""
JMA Unified Hypocenter Catalogue downloader and parser.

Uses the open Japan Meteorological Agency (JMA) Unified
Hypocenter Bulletin instead of the authenticated Hi-net portal.

JMA provides yearly hypocenter files as ZIP archives containing
fixed-width 96-byte hypocenter records.

Official source:
https://www.data.jma.go.jp/eqev/data/bulletin/hypo_e.html
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from zipfile import ZipFile

import requests

from eq_toolkit.catalog.model import Catalog


class JMA:
    """
    Downloader and parser for the public JMA Unified Hypocenter
    Catalogue.

    No JMA/Hi-net account or credentials are required.
    """

    BASE_URL = (
        "https://www.data.jma.go.jp/"
        "eqev/data/bulletin/data/hypo/"
    )

    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "ETAS-Toolkit/1.0 "
                    "(earthquake catalogue research)"
                )
            }
        )

    # ------------------------------------------------------------------
    # DOWNLOAD PUBLIC JMA YEARLY FILE
    # ------------------------------------------------------------------

    def download_year(self, year: int) -> bytes:
        """
        Download the official public JMA hypocenter ZIP file.

        Parameters
        ----------
        year : int
            Four-digit year.

        Returns
        -------
        bytes
            ZIP archive containing the JMA fixed-width catalogue.
        """

        year = int(year)

        if year < 1919:
            raise ValueError(
                "JMA public hypocenter catalogue starts from 1919."
            )

        url = f"{self.BASE_URL}h{year}.zip"

        response = self.session.get(
            url,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.content

    # ------------------------------------------------------------------
    # EXTRACT TEXT FROM JMA ZIP
    # ------------------------------------------------------------------

    def download_year_text(self, year: int) -> str:
        """
        Download a yearly JMA ZIP archive and extract its catalogue
        text.

        The JMA yearly files are ZIP archives containing fixed-width
        hypocenter records.
        """

        content = self.download_year(year)

        with ZipFile(BytesIO(content)) as archive:

            files = [
                name
                for name in archive.namelist()
                if not name.endswith("/")
            ]

            if not files:
                raise RuntimeError(
                    f"No data file found inside JMA {year} ZIP archive."
                )

            # Usually there is one catalogue file. Select the first
            # non-directory file.
            filename = files[0]

            raw = archive.read(filename)

        # JMA catalogue is essentially ASCII/fixed-width text.
        # Decode robustly so malformed bytes do not destroy the
        # complete download.
        return raw.decode("ascii", errors="replace")

    # ------------------------------------------------------------------
    # PARSE ONE 96-CHARACTER HYPOCENTER RECORD
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_record(line: str):
        """
        Parse one JMA 96-character hypocenter record.

        JMA record format:
            01       agency
            02-05    year
            06-07    month
            08-09    day
            10-11    hour
            12-13    minute
            14-17    seconds
            22-24    latitude degrees
            25-28    latitude minutes
            33-36    longitude degrees
            37-40    longitude minutes
            45-49    depth
            53-54    magnitude
            55       magnitude type
        """

        if len(line) < 96:
            return None

        agency_code = line[0]

        # J = JMA
        # U = USGS
        # I = other international organization
        #
        # The public JMA hypocenter bulletin can contain these
        # agency-coded hypocenter records.
        if agency_code not in {"J", "U", "I"}:
            return None

        try:
            year = int(line[1:5])
            month = int(line[5:7])
            day = int(line[7:9])
            hour = int(line[9:11])
            minute = int(line[11:13])

            second_text = line[13:17].strip()

            if not second_text:
                return None

            second = float(second_text) / 100.0

            # ------------------------------------------------------
            # LATITUDE
            # ------------------------------------------------------

            lat_deg_text = line[21:24].strip()
            lat_min_text = line[24:28].strip()

            if not lat_deg_text or not lat_min_text:
                return None

            latitude = (
                float(lat_deg_text)
                + float(lat_min_text) / 100.0 / 60.0
            )

            # ------------------------------------------------------
            # LONGITUDE
            # ------------------------------------------------------

            lon_deg_text = line[32:36].strip()
            lon_min_text = line[36:40].strip()

            if not lon_deg_text or not lon_min_text:
                return None

            longitude = (
                float(lon_deg_text)
                + float(lon_min_text) / 100.0 / 60.0
            )

            # ------------------------------------------------------
            # DEPTH
            # ------------------------------------------------------

            depth_text = line[44:49].strip()

            if not depth_text:
                return None

            depth = float(depth_text) / 100.0

            # ------------------------------------------------------
            # MAGNITUDE
            # ------------------------------------------------------

            magnitude_text = line[52:54].strip()
            magnitude_type = line[54:55].strip()

            if not magnitude_text:
                return None

            magnitude = JMA._parse_magnitude(magnitude_text)

            if magnitude is None:
                return None

            # ------------------------------------------------------
            # JMA TIME IS JST
            # ------------------------------------------------------

            whole_second = int(second)

            microsecond = int(
                round(
                    (second - whole_second) * 1_000_000
                )
            )

            # Protect against rounding 59.999999 -> 60.
            if microsecond >= 1_000_000:
                whole_second += 1
                microsecond -= 1_000_000

            time_jst = datetime(
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                second=whole_second,
                microsecond=microsecond,
                tzinfo=timezone(timedelta(hours=9)),
            )

            # Convert JMA/JST -> UTC.
            event_time = time_jst.astimezone(timezone.utc)

            # ------------------------------------------------------
            # EVENT ID
            # ------------------------------------------------------

            event_id = (
                f"JMA_{event_time.strftime('%Y%m%d%H%M%S')}"
                f"_{latitude:.4f}_{longitude:.4f}"
            )

            source_agency = {
                "J": "JMA",
                "U": "USGS",
                "I": "ISC",
            }[agency_code]

            return {
                "time": event_time,
                "latitude": latitude,
                "longitude": longitude,
                "depth": depth,
                "magnitude": magnitude,
                "magnitude_type": magnitude_type,
                "source_agency": source_agency,
                "event_id": event_id,
            }

        except (
            ValueError,
            TypeError,
            OverflowError,
        ):
            return None

    # ------------------------------------------------------------------
    # MAGNITUDE DECODER
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_magnitude(text: str):
        """
        Decode the JMA magnitude field.

        Normal positive magnitudes are stored as one decimal place.

        JMA also uses special representations for negative magnitudes:
            -0.1 -> -1
            ...
            -0.9 -> -9
            -1.0 -> A0
            -1.9 -> A9
            -2.0 -> B0
            -2.9 -> B9
            -3.0 -> C0
        """

        text = text.strip()

        if not text:
            return None

        # Normal magnitude.
        try:
            return float(text) / 10.0
        except ValueError:
            pass

        # Special negative-magnitude encoding.
        if len(text) == 2 and text[0].isalpha():
            letter = text[0].upper()
            digit = text[1]

            if digit.isdigit():
                base = {
                    "A": -1.0,
                    "B": -2.0,
                    "C": -3.0,
                }.get(letter)

                if base is not None:
                    return base - int(digit) / 10.0

        # -0.1 ... -0.9 encoded as -1 ... -9.
        if text.startswith("-") and len(text) == 2:
            try:
                digit = int(text[1])
                return -digit / 10.0
            except ValueError:
                return None

        return None

    # ------------------------------------------------------------------
    # PARSE COMPLETE DECK
    # ------------------------------------------------------------------

    def parse_deck(self, text: str) -> Catalog:
        """
        Parse complete JMA fixed-width catalogue text.

        Parameters
        ----------
        text : str
            JMA hypocenter catalogue text.

        Returns
        -------
        Catalog
            Normalized ETAS-Toolkit Catalog object.
        """

        catalog = Catalog()

        for line in text.splitlines():

            # IMPORTANT:
            # Keep trailing spaces because the JMA format is fixed-width.
            line = line.rstrip("\r\n")

            record = self._parse_record(line)

            if record is None:
                continue

            catalog.add_event(**record)

        return catalog

    # ------------------------------------------------------------------
    # FILTER AFTER PARSING
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_catalog(
        catalog: Catalog,
        starttime=None,
        endtime=None,
        min_magnitude=None,
        bbox=None,
    ) -> Catalog:
        """
        Apply catalogue filters after parsing.

        bbox:
            (west, south, east, north)

        IMPORTANT:
        Filtering does NOT redefine the observation origin.

        The observation origin must be supplied by the caller as
        the original observation-window start.
        """

        df = catalog.data.copy()

        if starttime is not None:
            starttime = JMA._to_utc_datetime(starttime)
            df = df[df["time"] >= starttime]

        if endtime is not None:
            endtime = JMA._to_utc_datetime(endtime)
            df = df[df["time"] <= endtime]

        if min_magnitude is not None:
            df = df[
                df["magnitude"] >= float(min_magnitude)
            ]

        if bbox is not None:
            west, south, east, north = bbox

            df = df[
                (df["longitude"] >= west)
                & (df["longitude"] <= east)
                & (df["latitude"] >= south)
                & (df["latitude"] <= north)
            ]

        new_catalog = Catalog()
        new_catalog.data = df.reset_index(drop=True)

        return new_catalog

    # ------------------------------------------------------------------
    # GET EVENTS
    # ------------------------------------------------------------------

    def get_events(
        self,
        deck_text=None,
        year=None,
        starttime=None,
        endtime=None,
        min_magnitude=None,
        bbox=None,
    ) -> Catalog:
        """
        Obtain JMA earthquake events.

        Two modes are supported.

        1. Parse an already downloaded deck:

            jma.get_events(deck_text=text)

        2. Download a public JMA yearly catalogue:

            jma.get_events(
                year=2020,
                starttime="2020-01-01T00:00:00",
                endtime="2020-12-31T23:59:59",
                min_magnitude=2.0,
            )

        No username or password is required.
        """

        if deck_text is not None:

            catalog = self.parse_deck(deck_text)

        elif year is not None:

            text = self.download_year_text(year)

            catalog = self.parse_deck(text)

        else:

            raise ValueError(
                "Provide either deck_text or year."
            )

        # Apply filters only after the complete public catalogue
        # has been parsed.
        if any(
            value is not None
            for value in (
                starttime,
                endtime,
                min_magnitude,
                bbox,
            )
        ):
            catalog = self._filter_catalog(
                catalog,
                starttime=starttime,
                endtime=endtime,
                min_magnitude=min_magnitude,
                bbox=bbox,
            )

        return catalog

    # ------------------------------------------------------------------
    # MULTI-YEAR DOWNLOAD
    # ------------------------------------------------------------------

    def get_events_range(
        self,
        starttime,
        endtime,
        min_magnitude=None,
        bbox=None,
    ) -> Catalog:
        """
        Download and combine all JMA yearly files required for an
        observation window.

        The observation window is preserved exactly.

        Example
        -------
        jma.get_events_range(
            "2020-01-01T00:00:00",
            "2022-12-31T23:59:59",
            min_magnitude=2.0,
        )
        """

        start = self._to_utc_datetime(starttime)
        end = self._to_utc_datetime(endtime)

        if end < start:
            raise ValueError(
                "endtime must be greater than or equal to starttime."
            )

        combined = Catalog()

        for year in range(start.year, end.year + 1):

            print(
                f"Downloading JMA Unified Hypocenter "
                f"Catalogue: {year}"
            )

            yearly = self.get_events(
                year=year,
                starttime=start,
                endtime=end,
                min_magnitude=min_magnitude,
                bbox=bbox,
            )

            combined.merge(yearly)

        # Remove duplicate records that can occur at year boundaries.
        if not combined.data.empty:
            combined.data = (
                combined.data
                .drop_duplicates(subset=["event_id"])
                .sort_values("time")
                .reset_index(drop=True)
            )

        return combined

    # ------------------------------------------------------------------
    # DATETIME NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def _to_utc_datetime(value) -> datetime:
        """
        Convert common datetime representations to timezone-aware UTC.
        """

        if isinstance(value, datetime):

            if value.tzinfo is None:
                return value.replace(
                    tzinfo=timezone.utc
                )

            return value.astimezone(timezone.utc)

        if isinstance(value, str):

            text = value.strip()

            if text.endswith("Z"):
                text = text[:-1] + "+00:00"

            dt = datetime.fromisoformat(text)

            if dt.tzinfo is None:
                dt = dt.replace(
                    tzinfo=timezone.utc
                )

            return dt.astimezone(timezone.utc)

        raise TypeError(
            "Datetime value must be a datetime object or ISO string."
        )