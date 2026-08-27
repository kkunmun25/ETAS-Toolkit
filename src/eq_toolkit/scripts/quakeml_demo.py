from eq_toolkit.catalog.quakeml import read_quakeml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
catalog_file = PROJECT_ROOT / "examples" / "sample_catalog.xml"
catalog = read_quakeml(catalog_file)

print(catalog.data.head())

catalog.save_csv("catalog_from_quakeml.csv")

print("QuakeML imported successfully!")