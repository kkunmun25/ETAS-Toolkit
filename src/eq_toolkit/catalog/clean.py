import pandas as pd
from eq_toolkit.catalog.model import Catalog


def drop_duplicates(catalog: Catalog) -> Catalog:
    new_catalog = Catalog()
    new_catalog.data = catalog.data.drop_duplicates().reset_index(drop=True)
    return new_catalog


def sort_by_time(catalog: Catalog) -> Catalog:
    new_catalog = Catalog()
    new_catalog.data = catalog.data.sort_values("time").reset_index(drop=True)
    return new_catalog


def filter_magnitude(catalog: Catalog, min_mag=None, max_mag=None) -> Catalog:
    df = catalog.data.copy()



    if min_mag is not None:
        df = df[df["magnitude"] >= min_mag]

    if max_mag is not None:
        df = df[df["magnitude"] <= max_mag]



    new_catalog = Catalog()
    new_catalog.data = df.reset_index(drop=True)
    return new_catalog


def filter_time(catalog: Catalog, start=None, end=None) -> Catalog:
    df = catalog.data.copy()

    if start is not None:
        df = df[df["time"] >= start]

    if end is not None:
        df = df[df["time"] <= end]

    new_catalog = Catalog()
    new_catalog.data = df.reset_index(drop=True)
    return new_catalog


def filter_region(
    catalog: Catalog,
    min_lon,
    max_lon,
    min_lat,
    max_lat,
) -> Catalog:

    df = catalog.data.copy()

    df = df[
        (df["longitude"] >= min_lon)
        & (df["longitude"] <= max_lon)
        & (df["latitude"] >= min_lat)
        & (df["latitude"] <= max_lat)
    ]

    new_catalog = Catalog()
    new_catalog.data = df.reset_index(drop=True)
    return new_catalog


def homogenize_magnitude(catalog: Catalog) -> Catalog:
    """
    Placeholder for future magnitude homogenization.
    """
    return catalog