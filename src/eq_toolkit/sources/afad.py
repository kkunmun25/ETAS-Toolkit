"""
eq_toolkit/sources/afad.py

Download earthquake catalogs from the AFAD (Turkey) API
and convert them into a Catalog object.
"""

import requests

from eq_toolkit.catalog.model import Catalog


class AFAD:
    """
    AFAD earthquake catalogue downloader.
    """

    BASE_URL = "https://deprem.afad.gov.tr/apiv2/event/filter"

    def __init__(self):
        pass

    def get_events(
        self,
        starttime,
        endtime,
        minlatitude,
        maxlatitude,
        minlongitude,
        maxlongitude,
        minmagnitude=0.0,
    ):

        params = {
            "start": starttime,
            "end": endtime,
            "minlat": minlatitude,
            "maxlat": maxlatitude,
            "minlon": minlongitude,
            "maxlon": maxlongitude,
            "minmag": minmagnitude,
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=60,
        )

        response.raise_for_status()

        events = response.json()

        catalog = Catalog()

        for event in events:

            catalog.add_event(
                time=event.get("date"),
                latitude=event.get("latitude"),
                longitude=event.get("longitude"),
                depth=event.get("depth"),
                magnitude=event.get("magnitude"),
                magnitude_type=event.get("type", "ML"),
                source_agency="AFAD",
                event_id=event.get("eventID"),
            )

        return catalog