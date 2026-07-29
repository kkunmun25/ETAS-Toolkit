from obspy import read_events

from eq_toolkit.catalog.model import Catalog


def read_quakeml(filename):
    events = read_events(filename)

    catalog = Catalog()

    for event in events:

        origin = event.preferred_origin() or event.origins[0]
        magnitude = event.preferred_magnitude() or event.magnitudes[0]

        catalog.add_event(
            time=origin.time.datetime,
            latitude=origin.latitude,
            longitude=origin.longitude,
            depth=origin.depth / 1000 if origin.depth else None,
            magnitude=magnitude.mag,
            magnitude_type=magnitude.magnitude_type,
            source_agency="USGS",
            event_id=str(event.resource_id),
        )

    return catalog