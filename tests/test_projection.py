from pyproj import Geod
from src.eq_toolkit.catalog.projection import lonlat_to_xy, xy_to_lonlat

# WGS84 is the standard Earth model used by GPS
geod = Geod(ellps="WGS84")


def test_round_trip():
    # Example coordinates (Bengaluru)
    lon = 77.5946
    lat = 12.9716

    # Convert to projected coordinates
    x, y = lonlat_to_xy(lon, lat)

    # Convert back to longitude and latitude
    lon2, lat2 = xy_to_lonlat(x, y)

    # Calculate the distance between the original and recovered points
    _, _, distance = geod.inv(lon, lat, lon2, lat2)

    print(f"Round-trip error = {distance:.6f} metres")

    # Professor's requirement: error must be less than 1 metre
    assert distance < 1.0