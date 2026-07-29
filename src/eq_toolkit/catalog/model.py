import pandas as pd


class Catalog:

    def __init__(self):
        print("A new Catalog has been created!")

        self.data = pd.DataFrame(
            columns=[
                "time",
                "latitude",
                "longitude",
                "depth",
                "magnitude",
                "magnitude_type",
                "source_agency",
                "event_id",
            ]
        )

    def add_event(self, time, latitude, longitude, depth, magnitude, magnitude_type, source_agency, event_id):
        new_event = pd.DataFrame(
            {
                "time": [time],
                "latitude": [latitude],
                "longitude": [longitude],
                "depth": [depth],
                "magnitude": [magnitude],
                "magnitude_type": [magnitude_type],
                "source_agency": [source_agency],
                "event_id": [event_id],
            }
        )
        self.data = pd.concat([self.data, new_event], ignore_index=True)

    def save_csv(self, filename):
        self.data.to_csv(filename, index=False)

    def load_csv(self, filename):
        self.data = pd.read_csv(filename)

    def save_parquet(self, filename):
        self.data.to_parquet(filename, index=False)

    def load_parquet(self, filename):
        self.data = pd.read_parquet(filename)

    @classmethod
    def from_geojson(cls, data):
        """
        Create a Catalog from an FDSN GeoJSON response.
        """

        catalog = cls()

        for feature in data["features"]:

            properties = feature["properties"]
            geometry = feature["geometry"]

            coordinates = geometry["coordinates"]

            catalog.add_event(
                time=properties["time"],
                latitude=coordinates[1],
                longitude=coordinates[0],
                depth=coordinates[2],
                magnitude=properties["mag"],
                magnitude_type=properties["magType"],
                source_agency=properties["net"],
                event_id=feature["id"],
            )

        return catalog

    def merge(self, other):
        """
        Merge another Catalog into this one.
        """
        
        self.data = pd.concat(
          [self.data, other.data],
          ignore_index=True
        )