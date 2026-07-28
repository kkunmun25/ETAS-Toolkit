from eq_toolkit.catalog.model import Catalog

catalog = Catalog()

catalog.add_event(
    time="2024-04-03T23:58:13.340Z",
    latitude=24.243,
    longitude=94.889,
    depth=134.2,
    magnitude=5.6,
    magnitude_type="mb",
    source_agency="USGS",
    event_id="us7000m9qd",
)

print(catalog.data)

catalog.save_csv("earthquakes.csv")
catalog.save_parquet("earthquakes.parquet")