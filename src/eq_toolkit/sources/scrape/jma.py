"""
JMA/NIED earthquake catalogue scraper.

Parses the fixed-width 96-byte JMA/NIED deck format
and provides an authenticated download interface.
"""

import os
from datetime import datetime, timedelta, timezone

import requests

from eq_toolkit.catalog.model import Catalog


class JMA:
    """
    Parser and downloader for the JMA/NIED earthquake catalogue.
    """

    LOGIN_URL = "https://hinetwww11.bosai.go.jp/auth/JMA/?LANG=en"

    def __init__(self, username=None, password=None):
        """
        Create a JMA scraper.

        Credentials are read from environment variables if they
        are not explicitly supplied.

        Environment variables:
            JMA_USERNAME
            JMA_PASSWORD
        """

        self.username = (
            username
            if username is not None
            else os.getenv("JMA_USERNAME")
        )

        self.password = (
            password
            if password is not None
            else os.getenv("JMA_PASSWORD")
        )

        self.session = requests.Session()

    # ---------------------------------------------------------
    # Parse fixed-width JMA/NIED deck file
    # ---------------------------------------------------------

    def parse_deck(self, text):
        """
        Parse JMA/NIED fixed-width 96-byte hypocenter records.

        Parameters
        ----------
        text : str
            JMA/NIED deck-format text.

        Returns
        -------
        Catalog
            Parsed earthquake catalogue.
        """

        catalog = Catalog()

        for line in text.splitlines():

            # Remove only newline characters.
            # DO NOT remove trailing spaces because this is
            # a fixed-width format.
            line = line.rstrip("\r\n")

            if not line:
                continue

            # JMA/NIED records are 96 characters long.
            if len(line) < 96:
                continue

            # Hypocenter records:
            # J = JMA
            # I = ISC
            # U = USGS
            if line[0] not in {"J", "I", "U"}:
                continue

            try:
               

                year = int(line[1:5])
                month = int(line[5:7])
                day = int(line[7:9])
                hour = int(line[9:11])
                minute = int(line[11:13])

                second_text = line[13:17].strip()

                if not second_text:
                    continue

              
                second = float(second_text) / 100.0

        
                # Latitude
            

                latitude_degree_text = line[21:24].strip()
                latitude_minute_text = line[24:28].strip()

                if not latitude_degree_text:
                    continue

                if not latitude_minute_text:
                    continue

                latitude_degree = float(latitude_degree_text)
                latitude_minute = (
                    float(latitude_minute_text) / 100.0
                )

                latitude = (
                    latitude_degree
                    + latitude_minute / 60.0
                )

                
                # Longitude
               

                longitude_degree_text = line[32:36].strip()
                longitude_minute_text = line[36:40].strip()

                if not longitude_degree_text:
                    continue

                if not longitude_minute_text:
                    continue

                longitude_degree = float(
                    longitude_degree_text
                )

                longitude_minute = (
                    float(longitude_minute_text) / 100.0
                )

                longitude = (
                    longitude_degree
                    + longitude_minute / 60.0
                )

            
                # Depth
            

                depth_text = line[44:49].strip()

                if not depth_text:
                    continue

                depth = float(depth_text) / 100.0

            
                # Magnitude
               
                magnitude_text = line[52:54].strip()
                magnitude_type = line[54:55].strip()

                if not magnitude_text:
                    continue

                magnitude = float(magnitude_text) / 10.0

                
                # Convert JMA time to UTC
            

                whole_second = int(second)
                microsecond = int(
                    round(
                        (second - whole_second)
                        * 1_000_000
                    )
                )

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

                time = time_jst.astimezone(
                    timezone.utc
                )

           
                # Event ID
               

                event_id = (
                    f"JMA_{time.strftime('%Y%m%d%H%M%S')}"
                )

               
                # Source agency
                

                if line[0] == "J":
                    source_agency = "JMA"
                elif line[0] == "I":
                    source_agency = "ISC"
                else:
                    source_agency = "USGS"

                # Add event
             

                catalog.add_event(
                    time=time,
                    latitude=latitude,
                    longitude=longitude,
                    depth=depth,
                    magnitude=magnitude,
                    magnitude_type=magnitude_type,
                    source_agency=source_agency,
                    event_id=event_id,
                )

            except (ValueError, TypeError, OverflowError):
                # Ignore malformed records.
                continue

        return catalog

    # Download
   

    def download_deck(self, download_url):
        """
        Download a JMA/NIED deck file.

        Parameters
        ----------
        download_url : str
            URL of the authenticated JMA/NIED download request.

        Returns
        -------
        str
            Downloaded deck-format text.

        Notes
        -----
        NIED requires registered-user authentication.
        The exact download request is created by the
        authenticated JMA page, so the URL should be supplied
        from the registered-user download page.

        Credentials are never hard-coded.
        """

        if not self.username or not self.password:
            raise RuntimeError(
                "JMA credentials are required. "
                "Set JMA_USERNAME and JMA_PASSWORD."
            )

        # NIED uses an authenticated web session.
        #
        # The exact download URL is generated by the
        # registered-user JMA page. We send the credentials
        # using HTTP authentication if the supplied endpoint
        # supports it.
        response = self.session.get(
            download_url,
            auth=(self.username, self.password),
            timeout=120,
        )

        response.raise_for_status()

        # Protect against accidentally receiving an HTML
        # login page instead of earthquake data.
        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if (
            "html" in content_type
            and not response.text.lstrip().startswith("J")
        ):
            raise RuntimeError(
                "JMA/NIED returned an HTML page instead "
                "of deck data. Authentication or the "
                "download URL may be incorrect."
            )

        return response.text

    # Get events
  

    def get_events(self, deck_text=None, download_url=None):
        """
        Obtain JMA earthquake events.

        Either provide deck_text directly or provide
        download_url for an authenticated download.

        """

        if deck_text is not None:
            return self.parse_deck(deck_text)

        if download_url is not None:
            text = self.download_deck(download_url)
            return self.parse_deck(text)

        raise ValueError(
            "Provide either deck_text or download_url."
        )