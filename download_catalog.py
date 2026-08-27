import requests
import pandas as pd

url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

params = {
    "format": "geojson",
    "starttime": "2016-01-01",
    "endtime": "2026-08-27",
    "minmagnitude": 3.0,
    "minlatitude": 5,
    "maxlatitude": 15,
    "minlongitude": 90,
    "maxlongitude": 100,
    "orderby": "time-asc",
    "limit": 20000,
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

rows = []

for feature in data["features"]:
    p = feature["properties"]
    coords = feature["geometry"]["coordinates"]

    rows.append({
        "time": pd.to_datetime(p["time"], unit="ms", utc=True),
        "latitude": coords[1],
        "longitude": coords[0],
        "depth": coords[2],
        "magnitude": p["mag"],
        "magnitude_type": p["magType"],
        "source_agency": p.get("net"),
        "event_id": feature["id"],
    })

df = pd.DataFrame(rows)

df = df.dropna(subset=["time", "latitude", "longitude", "magnitude"])

df.to_csv("andaman_catalog.csv", index=False)

print("Number of events:", len(df))
print(df.head())
print("\nSaved as: andaman_catalog.csv")