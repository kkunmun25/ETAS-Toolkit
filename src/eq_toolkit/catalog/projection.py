from pyproj import Transformer

# Geographic (longitude, latitude) -> Projected (x, y)
_to_xy = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:3857",
    always_xy=True,
)

# Projected (x, y) -> Geographic (longitude, latitude)
_to_lonlat = Transformer.from_crs(
    "EPSG:3857",
    "EPSG:4326",
    always_xy=True,
)

def lonlat_to_xy(lon: float, lat: float) -> tuple[float, float]:
    """
    Convert longitude and latitude (degrees)
    to projected x and y coordinates (metres).
    """
    x, y = _to_xy.transform(lon, lat)
    return x, y

def xy_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """
    Convert projected x and y coordinates (metres)
    to longitude and latitude (degrees).
    """
    lon, lat = _to_lonlat.transform(x, y)
    return lon, lat