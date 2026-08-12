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

    def _find_records(self, obj):
        """Find earthquake records anywhere inside the JSON response."""

        records = []

        if isinstance(obj, dict):

            if "OriginTime" in obj:
                records.append(obj)

            else:
                for value in obj.values():
                    records.extend(self._find_records(value))

        elif isinstance(obj, list):

            for item in obj:
                records.extend(self._find_records(item))

        return records

    def get_events(self, min_mag=None):
        """Download and return CWA earthquake events."""

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

        # Find earthquake records automatically
        records = self._find_records(data)

        rows = []

        for event in records:

            try:
                magnitude = float(event["LocalMagnitude"])

                if min_mag is not None and magnitude < min_mag:
                    continue

                rows.append(
                    {
                        "time": pd.to_datetime(
                            event["OriginTime"]
                        ),
                        "latitude": float(
                            event["EpicenterLatitude"]
                        ),
                        "longitude": float(
                            event["EpicenterLongitude"]
                        ),
                        "depth": float(
                            event["FocalDepth"]
                        ),
                        "magnitude": magnitude,
                        "magnitude_type": "ML",
                        "source_agency": self.source_agency,
                        "event_id": (
                            f"CWA-{event['OriginTime']}"
                        ),
                    }
                )

            except (KeyError, TypeError, ValueError):
                continue

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

        print("Records found:", len(records))
        print("Rows parsed:", len(df))

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

    catalog = CWA_Taiwan().get_events()

    print("Number of events:", len(catalog.data))
    print(catalog.data.head())