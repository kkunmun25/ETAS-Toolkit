"""
eq_toolkit/sources/scrape/koeri.py

Scraper for the KOERI earthquake catalogue.
"""

import requests
import pandas as pd

from eq_toolkit.catalog.model import Catalog


class KOERI:
    """KOERI earthquake catalogue scraper."""

    BASE_URL = "https://www.koeri.boun.edu.tr/sismo/2/en/"

    def __init__(self):
        pass

    def get_events(self, url=None):
        """
        Download the KOERI earthquake catalogue.
        """

        if url is None:
            url = self.BASE_URL

        response = requests.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        return self.parse(response.text)

    def parse(self, text):
        """
        Parse KOERI catalogue text.

        Parameters
        ----------
        text : str
            Raw catalogue text.

        Returns
        -------
        Catalog
            Parsed earthquake catalogue.
        """

        print("Parsing KOERI data...")

    

        columns = [
            "time",
            "latitude",
            "longitude",
            "depth",
            "magnitude",
            "magnitude_type",
            "source_agency",
            "event_id",
        ]

        df = pd.DataFrame(columns=columns)

        catalog = Catalog()

        return catalog