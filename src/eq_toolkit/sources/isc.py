import io

import pandas as pd
import requests

from eq_toolkit import catalog
from eq_toolkit.catalog.model import Catalog

ISC_URL = "https://www.isc.ac.uk/fdsnws/event/1/query"

def get_events(
    starttime,
    endtime,
    minmagnitude=None,
    maxmagnitude=None,
    minlatitude=None,
    maxlatitude=None,
    minlongitude=None,
    maxlongitude=None,
):
    """
    Get events from the ISC API.
    """

    params = {
        "format": "text",
        "starttime": starttime,
        "endtime": endtime,
    }

    if minmagnitude is not None:
        params["minmagnitude"] = minmagnitude
    if maxmagnitude is not None:
        params["maxmagnitude"] = maxmagnitude
    if minlatitude is not None:
        params["minlatitude"] = minlatitude
    if maxlatitude is not None:
        params["maxlatitude"] = maxlatitude
    if minlongitude is not None:
        params["minlongitude"] = minlongitude
    if maxlongitude is not None:
        params["maxlongitude"] = maxlongitude

    response = requests.get(ISC_URL, params=params ,timeout=30 ,)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text),sep="|",)

    catalog = Catalog()

    for _, row in df.iterrows():
     catalog.add_event(
        time=row["Time"],
        latitude=row["Latitude"],
        longitude=row["Longitude"],
        depth=row["Depth/km"],
        magnitude=row["Magnitude"],
        magnitude_type=row["MagType"],
        source_agency=row["Author"],
        event_id=row["#EventID"],
    )

    return catalog