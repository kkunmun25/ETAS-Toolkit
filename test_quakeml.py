from eq_toolkit.catalog.quakeml import read_quakeml

catalog = read_quakeml(r"C:\Users\tapas\Downloads\sample_catalog.xml")

print(catalog.data.head())

catalog.save_csv("catalog_from_quakeml.csv")

print("QuakeML imported successfully!")