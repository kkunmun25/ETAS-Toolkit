import io
import zipfile

import pandas as pd
import requests
import urllib3

from ...catalog.model import Catalog

urllib3.disable_warnings()


class CSN_Chile:
    """Download earthquake catalog from Chile CSN."""

    BASE_URL = "https://www.sismologia.cl/"
    ZENODO_URL = "https://zenodo.org/api/records/11360590"

    def __init__(self):
        self.catalog = None

    def download(self):
        """Download the CSN Chile webpage."""
        response = requests.get(
            self.BASE_URL,
            timeout=30,
            verify=False,
        )
        response.raise_for_status()
        return response.text

    def _find_earthquake_table(self, html):
        """Find the most likely earthquake table on the CSN webpage."""
        tables = pd.read_html(io.StringIO(html))

        if not tables:
            raise ValueError("No tables found on CSN Chile webpage.")

        # Look for a table containing earthquake-related columns.
        keywords = [
            "magnitud",
            "magnitude",
            "latitud",
            "latitude",
            "longitud",
            "longitude",
            "profundidad",
            "depth",
        ]

        for table in tables:
            columns = " ".join(
                str(column).lower()
                for column in table.columns
            )

            if any(keyword in columns for keyword in keywords):
                return table

        # If no obvious table is found, use the first table.
        return tables[0]

    def parse(self, html):
        """Parse earthquake data from CSN HTML."""
        df = self._find_earthquake_table(html)

        # Flatten MultiIndex columns if present.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                " ".join(str(x) for x in column).strip()
                for column in df.columns
            ]

        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

        return df

    def _find_column(self, df, possible_names):
        """Find a column using several possible names."""
        for column in df.columns:
            column_text = str(column).lower()

            for name in possible_names:
                if name in column_text:
                    return column

        return None

    def standardize(self, df):
        """Convert CSN columns to the standard catalog format."""

        date_col = self._find_column(
            df,
            ["fecha", "date"],
        )

        time_col = self._find_column(
            df,
            ["hora", "time"],
        )

        lat_col = self._find_column(
            df,
            ["latitud", "latitude", "lat"],
        )

        lon_col = self._find_column(
            df,
            ["longitud", "longitude", "lon", "lng"],
        )

        depth_col = self._find_column(
            df,
            ["profundidad", "depth"],
        )

        magnitude_col = self._find_column(
            df,
            ["magnitud", "magnitude", "mag"],
        )

        if magnitude_col is None:
            raise ValueError(
                "Could not find magnitude column in CSN data."
            )

        result = pd.DataFrame(index=df.index)

        # Time
        if date_col is not None and time_col is not None:
            result["time"] = pd.to_datetime(
                df[date_col].astype(str)
                + " "
                + df[time_col].astype(str),
                errors="coerce",
            )

        elif date_col is not None:
            result["time"] = pd.to_datetime(
                df[date_col],
                errors="coerce",
            )

        else:
            result["time"] = pd.NaT

        # Latitude
        if lat_col is not None:
            result["latitude"] = pd.to_numeric(
                df[lat_col],
                errors="coerce",
            )
        else:
            result["latitude"] = float("nan")

        # Longitude
        if lon_col is not None:
            result["longitude"] = pd.to_numeric(
                df[lon_col],
                errors="coerce",
            )
        else:
            result["longitude"] = float("nan")

        # Depth
        if depth_col is not None:
            result["depth"] = pd.to_numeric(
                df[depth_col],
                errors="coerce",
            )
        else:
            result["depth"] = float("nan")

        # Magnitude
        result["magnitude"] = pd.to_numeric(
            df[magnitude_col],
            errors="coerce",
        )

        # Additional standard fields
        result["magnitude_type"] = None
        result["source_agency"] = "CSN"

        result["event_id"] = [
            f"CSN_{i}"
            for i in range(len(result))
        ]

        # Remove rows where there is no magnitude.
        result = result.dropna(
            subset=["magnitude"]
        ).reset_index(drop=True)

        return result[
            [
                "time",
                "latitude",
                "longitude",
                "depth",
                "magnitude",
                "magnitude_type",
                "source_agency",
                "event_id",
            ]
        ]

    def _download_zenodo(self):
        """Download the static CSN catalog from Zenodo."""

        response = requests.get(
            self.ZENODO_URL,
            timeout=30,
        )
        response.raise_for_status()

        record = response.json()

        files = record.get("files", [])

        if not files:
            raise ValueError(
                "No files found in Zenodo record."
            )

        # Download the first suitable file.
        for file_info in files:
            file_url = file_info["links"]["self"]
            filename = file_info["key"]

            response = requests.get(
                file_url,
                timeout=120,
            )
            response.raise_for_status()

            print(
                f"Downloaded Zenodo file: {filename}"
            )

            return filename, response.content

        raise ValueError(
            "Could not download Zenodo catalog."
        )

    def _parse_zenodo(self):
        """Parse the Zenodo fallback catalog."""

        filename, content = self._download_zenodo()

        lower_name = filename.lower()

        # CSV
        if lower_name.endswith(".csv"):
            return pd.read_csv(
                io.BytesIO(content)
            )

        # TXT
        if lower_name.endswith(".txt"):
            return pd.read_csv(
                io.BytesIO(content),
                sep=None,
                engine="python",
            )

        # ZIP
        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(
                io.BytesIO(content)
            ) as z:
                names = z.namelist()

                print("Files inside Zenodo ZIP:")
                for name in names:
                    print(name)

                for name in names:
                    lower = name.lower()

                    if lower.endswith(
                        (".csv", ".txt", ".dat")
                    ):
                        with z.open(name) as file:
                            if lower.endswith(".csv"):
                                return pd.read_csv(file)

                            return pd.read_csv(
                                file,
                                sep=None,
                                engine="python",
                            )

        raise ValueError(
            f"Unsupported Zenodo file: {filename}"
        )

    def get_dataframe(self):
        """
        Get CSN earthquake data.

        First tries the CSN webpage.
        If that fails, uses the Zenodo catalog.
        """

        try:
            print("Downloading CSN Chile webpage...")

            html = self.download()

            df = self.parse(html)

            print(
                f"CSN HTML table found: {len(df)} rows"
            )

            return self.standardize(df)

        except Exception as exc:
            print(
                "CSN HTML scraping failed."
            )
            print(f"Reason: {exc}")
            print(
                "Trying Zenodo fallback..."
            )

            df = self._parse_zenodo()

            return self.standardize(df)

    def get_catalog(self):
        """Return earthquake data as a Catalog."""

        df = self.get_dataframe()

        print(
            f"Standardized CSN events: {len(df)}"
        )

        

        try:
            self.catalog = Catalog(data=df)
        except TypeError:
            try:
                self.catalog = Catalog(df=df)
            except TypeError:
            
                self.catalog = Catalog()
                self.catalog.data = df

        return self.catalog


if __name__ == "__main__":
    scraper = CSN_Chile()

    catalog = scraper.get_catalog()

    print(catalog)