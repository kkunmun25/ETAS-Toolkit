"""
eq_toolkit/sources/scrape/koeri.py

Scraper for the KOERI earthquake catalogue.
"""

import requests
import pandas as pd

from eq_toolkit.catalog.model import Catalog


class KOERI:
   

    BASE_URL = "https://www.koeri.boun.edu.tr/sismo/2/en/"

    def __init__(self):
        pass

    def get_events(self, url=None):
        

        if url is None:
            url = self.BASE_URL

        # Download the webpage
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Get webpage text
        text = response.text

        print("KOERI page downloaded successfully.")
        print(text[:1000])

        # Creating an empty dataframe with the standard Catalog columns
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

        # Create Catalog
        catalog = Catalog(df)

        return catalog