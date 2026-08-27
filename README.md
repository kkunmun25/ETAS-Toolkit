# ETAS Toolkit

A reproducible Python toolkit for **earthquake catalog processing, statistical analysis, visualization, and ETAS-oriented seismicity workflows**.

The project is designed to provide a modular and testable workflow for working with earthquake catalogs obtained from local files and FDSN-compatible services.

---

## Features

* Earthquake catalog ingestion and management
* QuakeML catalog reading and CSV export
* Catalog cleaning and quality control
* FDSN/USGS earthquake catalog retrieval
* Automatic handling of large catalog requests and API limits
* Geographic coordinate projection utilities
* Gutenberg–Richter frequency–magnitude analysis
* Magnitude of completeness (`Mc`) estimation
* Inter-event time analysis
* Earthquake declustering
* Exploratory data analysis and visualization
* Spatial and temporal earthquake-density analysis
* ETAS-related catalog preparation and analysis
* Reproducible examples and automated testing

---

## Project Structure

```text
ETAS-Toolkit/
│
├── src/
│   └── eq_toolkit/
│       ├── catalog/
│       │   ├── model.py
│       │   ├── clean.py
│       │   ├── fdsn.py
│       │   ├── quakeml.py
│       │   └── projection.py
│       │
│       ├── analysis/
│       │   └── ...
│       │
│       └── viz/
│           └── ...
│
├── tests/
│   ├── test_clean.py
│   ├── test_fdsn.py
│   ├── test_model.py
│   ├── test_projection.py
│   ├── test_quakeml_*.py
│   ├── test_fmd_*.py
│   ├── test_eda_plot.py
│   └── ...
│
├── examples/
│   ├── sample_catalog.xml
│   ├── mc_spatial_map.py
│   └── run_real_sc_mc.py
│
├── docs/
│
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kkunmun25/ETAS-Toolkit.git
cd ETAS-Toolkit
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

Install testing dependencies:

```bash
pip install pytest
```

---

## Basic Usage

The toolkit can be imported directly from Python:

```python
from eq_toolkit.catalog.model import Catalog
```

### Reading a QuakeML catalog

Example input:

```text
examples/sample_catalog.xml
```

A QuakeML catalog can be read using:

```python
from pathlib import Path
from eq_toolkit.catalog.quakeml import read_quakeml

catalog_file = Path("examples/sample_catalog.xml")
catalog = read_quakeml(catalog_file)

print(catalog.data.head())
```

---

## Earthquake Catalog Processing

The catalog workflow supports:

1. Catalog ingestion
2. Validation and cleaning
3. Magnitude and temporal filtering
4. Geographic processing
5. Statistical analysis
6. Visualization
7. Export to common tabular formats

The `Catalog` class provides a common representation for earthquake-event data throughout the workflow.

---

## FDSN Catalog Retrieval

The toolkit provides an FDSN client for retrieving earthquake catalogs from compatible services.

The implementation supports:

* Geographic bounding boxes
* Time-window queries
* Magnitude filtering
* Large catalog requests
* Automatic request windowing
* Handling of service/event limits

This allows large earthquake catalogs to be retrieved without relying on a single oversized API request.

---

## Statistical Analysis

### Gutenberg–Richter Relation

The toolkit supports frequency–magnitude analysis based on the Gutenberg–Richter relationship:

$$
\log_{10}N(M) = a-bM
$$

where:

* `N(M)` is the number of earthquakes with magnitude ≥ `M`
* `a` represents overall seismicity level
* `b` represents the relative proportion of small and large earthquakes

### Magnitude of Completeness

Several approaches are implemented for estimating the magnitude of completeness (`Mc`), including:

* Maximum Curvature (MAXC)
* Goodness-of-Fit (GFT)
* B-value Stability / related methods
* EMR-based estimation
* MBASS

These methods can be compared to evaluate the completeness of an earthquake catalog.

---

## Earthquake Declustering

The toolkit includes earthquake declustering functionality to distinguish background seismicity from clustered earthquake activity.

This provides a basis for comparing:

* Raw earthquake catalogs
* Declustered catalogs
* Background seismicity

Declustering is particularly relevant for subsequent seismicity-rate and ETAS-oriented analyses.

---

## Visualization

The visualization modules provide tools for exploring earthquake catalogs through:

* Frequency–magnitude distributions
* Spatial earthquake maps
* Temporal distributions
* Inter-event time plots
* Exploratory data analysis
* Spatial density analysis
* Magnitude-related visualizations

Example scripts are provided in the `examples/` directory.

---

## Testing

The repository contains an automated test suite covering catalog handling, cleaning, projections, FDSN functionality, statistical analysis, and visualization-related components.

Run the complete test suite from the project root:

```bash
pytest tests/
```

To verify test discovery without executing the tests:

```bash
pytest tests/ --collect-only -q
```

The tests are designed to be independent of machine-specific file paths.

Example data required by demonstrations are stored inside the repository rather than referenced through local paths such as:

```text
C:\Users\...
```

or:

```text
/content/
```

---

## Reproducibility

The project follows a reproducible workflow:

* Source code is maintained under `src/`
* Tests are maintained under `tests/`
* Demonstration inputs and scripts are maintained under `examples/`
* Dependencies are specified in `pyproject.toml`
* Tests can be executed from a clean repository clone
* Machine-specific absolute paths are avoided

---

## Example

A simple catalog workflow:

```python
from pathlib import Path
from eq_toolkit.catalog.quakeml import read_quakeml

catalog = read_quakeml(
    Path("examples/sample_catalog.xml")
)

print(catalog.data.head())
print(f"Number of events: {len(catalog.data)}")
```

---

## Data

The repository contains example earthquake catalog data used for development, analysis, and demonstrations.

Large external earthquake catalogs should be treated as input datasets rather than hard-coded dependencies of the Python package.

---

## Development

For development, install the package in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
pytest tests/
```

Check test discovery:

```bash
pytest tests/ --collect-only -q
```

---

## Project Goals

The main goals of the project are to provide:

* **Reproducibility** — workflows that can be reproduced from a clean clone
* **Modularity** — separate components for catalog handling, analysis, and visualization
* **Testability** — automated tests for core functionality
* **Scalability** — support for large earthquake catalog retrieval
* **Extensibility** — a foundation for future ETAS modelling and forecasting workflows



---

## License

This project is intended for academic and research use.


