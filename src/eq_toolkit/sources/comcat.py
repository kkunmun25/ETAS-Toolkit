"""
USGS ComCat earthquake catalog downloader.

Provides a normalized get_events() interface and automatically
splits large time windows when the USGS service rejects a query
because it contains too many events.
"""

from datetime import datetime, timedelta

import requests

from eq_toolkit.catalog.model import Catalog


COMCAT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# USGS ComCat maximum number of events returned by one request.
MAX_EVENTS = 20000


def _download_events(
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
    Download one time window from USGS ComCat.

    Returns
    -------
    list
        Raw GeoJSON earthquake features.
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

    response = requests.get(
        COMCAT_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    return data["features"]


def _collect_window(
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
    Download a time window.

    If USGS rejects the query because it is too large,
    split the time window into two smaller windows recursively.
    """

    print(
        f"Downloading: {starttime} -> {endtime}"
    )

    try:
        features = _download_events(
            starttime=starttime,
            endtime=endtime,
            minmagnitude=minmagnitude,
            maxmagnitude=maxmagnitude,
            minlatitude=minlatitude,
            maxlatitude=maxlatitude,
            minlongitude=minlongitude,
            maxlongitude=maxlongitude,
        )

    
        if len(features) >= MAX_EVENTS:

            print(
                f"Received {len(features)} events. "
                "Splitting time window..."
            )

            return _split_window(
                starttime=starttime,
                endtime=endtime,
                minmagnitude=minmagnitude,
                maxmagnitude=maxmagnitude,
                minlatitude=minlatitude,
                maxlatitude=maxlatitude,
                minlongitude=minlongitude,
                maxlongitude=maxlongitude,
            )

        return features

    except requests.HTTPError as exc:

        # USGS commonly returns HTTP 400 for a query that is
        # too large.
        if exc.response is not None and exc.response.status_code == 400:

            print(
                "USGS rejected the query. "
                "Splitting the time window..."
            )

            return _split_window(
                starttime=starttime,
                endtime=endtime,
                minmagnitude=minmagnitude,
                maxmagnitude=maxmagnitude,
                minlatitude=minlatitude,
                maxlatitude=maxlatitude,
                minlongitude=minlongitude,
                maxlongitude=maxlongitude,
            )

        # Any other HTTP error should be reported normally.
        raise


def _split_window(
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
    Split a time window into two smaller windows.
    """

    start = datetime.fromisoformat(
        starttime.replace("Z", "")
    )

    end = datetime.fromisoformat(
        endtime.replace("Z", "")
    )

    # Make sure the window can actually be split.
    if end <= start:
        raise RuntimeError(
            "Unable to split ComCat query further."
        )

    midpoint = start + (end - start) / 2

    first_start = start
    first_end = midpoint

    second_start = midpoint + timedelta(seconds=1)
    second_end = end

    first_features = _collect_window(
        starttime=first_start.isoformat(),
        endtime=first_end.isoformat(),
        minmagnitude=minmagnitude,
        maxmagnitude=maxmagnitude,
        minlatitude=minlatitude,
        maxlatitude=maxlatitude,
        minlongitude=minlongitude,
        maxlongitude=maxlongitude,
    )

    second_features = _collect_window(
        starttime=second_start.isoformat(),
        endtime=second_end.isoformat(),
        minmagnitude=minmagnitude,
        maxmagnitude=maxmagnitude,
        minlatitude=minlatitude,
        maxlatitude=maxlatitude,
        minlongitude=minlongitude,
        maxlongitude=maxlongitude,
    )

    return first_features + second_features


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
    Get earthquake events from the USGS ComCat API.

    Large queries are automatically divided into smaller
    time windows when necessary.

    """

    features = _collect_window(
        starttime=starttime,
        endtime=endtime,
        minmagnitude=minmagnitude,
        maxmagnitude=maxmagnitude,
        minlatitude=minlatitude,
        maxlatitude=maxlatitude,
        minlongitude=minlongitude,
        maxlongitude=maxlongitude,
    )

    catalog = Catalog()
    seen_ids = set()

    for event in features:

        event_id = event["id"]

        if event_id in seen_ids:
            continue

        seen_ids.add(event_id)

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
            event_id=event_id,
        )

    return catalog