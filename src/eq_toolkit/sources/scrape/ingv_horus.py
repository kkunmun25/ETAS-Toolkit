"""
INGV HORUS earthquake catalogue scraper.
"""

import io
import zipfile

import pandas as pd
import requests

from eq_toolkit.catalog.model import Catalog


class INGVHORUS:
    """
    Download earthquake data from INGV HORUS.
    """

    CATALOG_URL = (
        "https://horus.bo.ingv.it/"
        "DataFolder/HORUS_Ita_Catalog.zip"
    )

    def __init__(self):
        pass

    def get_events(self):
        """Download and parse the HORUS catalogue."""

        # Download the ZIP file
        response = requests.get(
            self.CATALOG_URL,
            timeout=120,
        )
        response.raise_for_status()

        print("Download successful!")

        # Open ZIP file
        zip_file = zipfile.ZipFile(
            io.BytesIO(response.content)
        )

        # Read the main HORUS catalogue
        catalog_text = zip_file.read(
            "HORUS_Ita_Catalog.txt"
        ).decode("utf-8")

        # Read whitespace-separated data
        df = pd.read_csv(
            io.StringIO(catalog_text),
            sep=r"\s+",
            skiprows=1,
            header=None,
            usecols=range(11),
        )

        # Give columns names
        df.columns = [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "latitude",
            "longitude",
            "depth",
            "magnitude",
            "sigma_magnitude",
        ]

        # Create datetime
        df["time"] = pd.to_datetime(
            {
                "year": df["year"].astype(int),
                "month": df["month"].astype(int),
                "day": df["day"].astype(int),
                "hour": df["hour"].astype(int),
                "minute": df["minute"].astype(int),
                "second": df["second"].astype(int),
            }
        )

        # Create dataframe in our standard format
        catalog_df = pd.DataFrame(
            {
                "time": df["time"],
                "latitude": df["latitude"],
                "longitude": df["longitude"],
                "depth": df["depth"],
                "magnitude": df["magnitude"],
                "magnitude_type": "Mw",
                "source_agency": "INGV-HORUS",
                "event_id": [
                    f"HORUS_{i}"
                    for i in range(len(df))
                ],
            }
        )

        print("Number of events:", len(catalog_df))
        print(catalog_df.head())

        # Create Catalog
        catalog = Catalog()

        # Add events
        for _, row in catalog_df.iterrows():
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