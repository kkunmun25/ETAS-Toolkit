import requests

from eq_toolkit.catalog.model import Catalog
COMCAT_URL = ("https://earthquake.usgs.gov/fdsnws/event/1/query")

def get_events(
    starttime,
    endtime,
    minmagnitude,
    maxmagnitude=None,
    minlatitude=None,
    maxlatitude=None,
    minlongitude=None,
    maxlongitude=None,
):
    """
    Get events from the ComCat API.
    """

    params = {
        "format": "geojson",
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": minmagnitude,
    }

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

    response = requests.get(COMCAT_URL, params=params)
    response.raise_for_status()

    data = response.json()
    catalog = Catalog()

    for event in data["features"]:
        properties = event["properties"]
        coordinates = event["geometry"]["coordinates"]

        catalog.add_event(
            time=properties["time"],
            latitude=coordinates[1],
            longitude=coordinates[0],
            depth=coordinates[2],
            magnitude=properties["mag"],
            magnitude_type=properties.get("magType"),
            source_agency="USGS",
            event_id=event["id"],
        )

    return catalog