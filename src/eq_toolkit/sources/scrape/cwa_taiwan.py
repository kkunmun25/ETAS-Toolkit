"""
CWA Taiwan earthquake catalog scraper.
"""

import requests
import pandas as pd
import urllib3

from ...catalog.model import Catalog


urllib3.disable_warnings()


class CWA_Taiwan:
    """Download earthquake catalog from CWA Taiwan."""

    BASE_URL = (
        "https://opendata.cwa.gov.tw/"
        "fileapi/v1/opendataapi/E-A0073-001"
    )

    def __init__(self):
        self.source_agency = "CWA"

    def get_events(self, min_mag=None):
        """Download and parse CWA earthquake catalog."""

        params = {
            "Authorization": "rdec-key-123-45678-011121314",
            "format": "JSON",
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=30,
            verify=False,
        )

        response.raise_for_status()

        data = response.json()
        # Extract records from response; structure may vary so try common keys
        dataset = data.get("cwaopendata", {}).get("Dataset", {})

        # dataset might be a list of records or a dict containing records under several possible keys
        if isinstance(dataset, list):
            records = dataset
        elif isinstance(dataset, dict):
            # try common keys
            records = (
                dataset.get("Records")
                or dataset.get("Data")
                or dataset.get("Record")
                or dataset.get("dataset")
                or []
            )
        else:
            records = []

        rows = []
        for event in records:
            try:
                rows.append(
                    {
                        "time": pd.to_datetime(event.get("OriginTime")),
                        "latitude": float(event.get("EpicenterLatitude", "nan")),
                        "longitude": float(event.get("EpicenterLongitude", "nan")),
                        "depth": float(event.get("FocalDepth", "nan")),
                        "magnitude": float(event.get("LocalMagnitude", "nan")),
                        "magnitude_type": event.get("MagnitudeType", "ML"),
                        "source_agency": self.source_agency,
                        "event_id": f"CWA-{event.get('OriginTime')}",
                    }
                )
            except Exception:
                # skip malformed records
                continue

        df = pd.DataFrame(rows)

        if min_mag is not None and not df.empty:
            df = df[df["magnitude"] >= min_mag]

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


if __name__ == "__main__":
    catalog = CWA_Taiwan().get_events(min_mag=4.0)

    print("Number of events:", len(catalog.data))
    print(catalog.data.head())  