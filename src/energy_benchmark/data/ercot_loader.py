"""ERCOT hourly load data downloader and parser.

Downloads historical hourly load data from ERCOT (Electric Reliability Council
of Texas) and produces a clean pandas Series with DatetimeIndex.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Known download URLs for ERCOT native load data (ZIP files containing Excel).
# These may change when ERCOT reorganises its website; update as needed.
ERCOT_URLS: Dict[int, str] = {
    2024: "https://www.ercot.com/files/docs/2025/01/10/Native_Load_2024.zip",
    2023: "https://www.ercot.com/files/docs/2024/01/08/Native_Load_2023.zip",
    2022: "https://www.ercot.com/files/docs/2023/01/31/Native_Load_2022.zip",
    2021: "https://www.ercot.com/files/docs/2022/01/07/Native_Load_2021.zip",
    2020: "https://www.ercot.com/files/docs/2021/01/12/Native_Load_2020.zip",
}

# Default target column (total ERCOT system load)
TARGET_COLUMN = "ERCOT"


class ERCOTLoader:
    """Download, cache and parse ERCOT hourly load data.

    Args:
        years: List of years to include. Defaults to 2020-2024.
        data_dir: Directory for caching downloaded files.
        target_column: Column to extract as the target series.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        years: Optional[List[int]] = None,
        data_dir: str | Path = "data/raw",
        target_column: str = TARGET_COLUMN,
        timeout: int = 120,
    ) -> None:
        self.years = years or list(range(2020, 2025))
        self.data_dir = Path(data_dir)
        self.target_column = target_column
        self.timeout = timeout

        # Validate requested years
        unsupported = set(self.years) - set(ERCOT_URLS.keys())
        if unsupported:
            raise ValueError(
                f"No known URL for year(s): {sorted(unsupported)}. "
                f"Supported years: {sorted(ERCOT_URLS.keys())}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, force_download: bool = False) -> pd.Series:
        """Download (if needed) and return the full hourly load series.

        Args:
            force_download: Re-download even if cached files exist.

        Returns:
            A pandas Series with DatetimeIndex (hourly frequency) containing
            system load in MW for the requested years, sorted chronologically.
        """
        frames: list[pd.DataFrame] = []
        for year in sorted(self.years):
            df = self._load_year(year, force_download=force_download)
            frames.append(df)

        combined = pd.concat(frames, axis=0)
        combined = combined.sort_index()

        # Remove any duplicate timestamps (DST transitions can cause this)
        combined = combined[~combined.index.duplicated(keep="first")]

        series = combined[self.target_column].rename("load_mw")

        logger.info(
            "Loaded ERCOT data: %d observations from %s to %s",
            len(series),
            series.index.min(),
            series.index.max(),
        )
        return series

    def split(
        self,
        series: pd.Series,
        train_end: str = "2022-12-31",
        val_end: str = "2023-06-30",
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Split series into train / validation / test sets.

        Args:
            series: Full load series from :meth:`load`.
            train_end: Last date (inclusive) of training set.
            val_end: Last date (inclusive) of validation set.
                     Everything after becomes the test set.

        Returns:
            (train, val, test) tuple of pandas Series.
        """
        train = series[series.index <= pd.Timestamp(train_end)]
        val = series[
            (series.index > pd.Timestamp(train_end))
            & (series.index <= pd.Timestamp(val_end))
        ]
        test = series[series.index > pd.Timestamp(val_end)]

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(train),
            len(val),
            len(test),
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_year(
        self, year: int, force_download: bool = False
    ) -> pd.DataFrame:
        """Load a single year, downloading the ZIP if necessary."""
        cache_path = self.data_dir / f"ercot_{year}.parquet"

        if cache_path.exists() and not force_download:
            logger.debug("Loading cached data for %d", year)
            return pd.read_parquet(cache_path)

        # Download and parse
        zip_bytes = self._download_zip(year)
        df = self._parse_zip(zip_bytes, year)

        # Cache as parquet for fast subsequent loads
        self.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached %d data to %s", year, cache_path)

        return df

    def _download_zip(self, year: int) -> bytes:
        """Download the ZIP file for a given year from ERCOT."""
        url = ERCOT_URLS[year]
        logger.info("Downloading ERCOT %d data from %s", year, url)

        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        logger.info(
            "Downloaded %.1f MB for %d",
            len(response.content) / 1e6,
            year,
        )
        return response.content

    def _parse_zip(self, zip_bytes: bytes, year: int) -> pd.DataFrame:
        """Extract and parse the Excel/CSV file inside the ERCOT ZIP.

        ERCOT ZIP files typically contain a single Excel file with columns:
        - ``Hour_Ending`` (timestamp string like "01/01/2024 01:00")
        - ``COAST``, ``EAST``, ``FAR_WEST``, ... (zone loads)
        - ``ERCOT`` (total system load)
        """
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Find the data file inside the ZIP
            names = zf.namelist()
            data_file = self._find_data_file(names)
            logger.debug("Parsing %s from ZIP", data_file)

            with zf.open(data_file) as f:
                content = io.BytesIO(f.read())
                if data_file.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(content)
                else:
                    df = pd.read_csv(content)

        df = self._normalise_columns(df, year)
        return df

    @staticmethod
    def _find_data_file(names: List[str]) -> str:
        """Pick the main data file from a list of ZIP entry names."""
        for name in names:
            lower = name.lower()
            # Skip hidden/metadata files
            if lower.startswith("__") or lower.startswith("."):
                continue
            if lower.endswith((".xlsx", ".xls", ".csv")):
                return name
        raise FileNotFoundError(
            f"No Excel/CSV file found in ZIP archive. Contents: {names}"
        )

    def _normalise_columns(
        self, df: pd.DataFrame, year: int
    ) -> pd.DataFrame:
        """Normalise column names, parse timestamps, set DatetimeIndex."""
        # Standardise column names: strip whitespace, uppercase
        df.columns = df.columns.str.strip()

        # Identify timestamp column (ERCOT uses varying names)
        ts_col = self._find_timestamp_column(df.columns.tolist())

        # Parse timestamps
        # ERCOT uses "Hour_Ending" format where "01:00" means the hour
        # ending at 01:00 (i.e. the 00:00-01:00 interval).
        # Some years use "24:00" for midnight, which pandas cannot parse
        # directly — we handle this by replacing "24:00" with "00:00" and
        # adding one day.
        ts_raw = df[ts_col].astype(str).str.strip()
        timestamps = self._parse_ercot_timestamps(ts_raw)

        # Build output dataframe with numeric load columns
        numeric_cols = [
            c for c in df.columns if c != ts_col and df[c].dtype.kind in "ifc"
        ]
        if self.target_column not in numeric_cols:
            # Try case-insensitive match
            mapping = {c.upper(): c for c in numeric_cols}
            if self.target_column.upper() in mapping:
                pass  # will work via original column name
            else:
                raise KeyError(
                    f"Target column '{self.target_column}' not found. "
                    f"Available numeric columns: {numeric_cols}"
                )

        out = df[numeric_cols].copy()
        out.index = timestamps
        out.index.name = "timestamp"

        # Drop rows where target is NaN
        out = out.dropna(subset=[self.target_column])

        return out

    @staticmethod
    def _find_timestamp_column(columns: List[str]) -> str:
        """Heuristically identify the timestamp column."""
        candidates = ["Hour_Ending", "HourEnding", "Hour Ending", "Datetime"]
        col_upper = {c.upper(): c for c in columns}
        for candidate in candidates:
            if candidate.upper() in col_upper:
                return col_upper[candidate.upper()]
        # Fallback: first column
        return columns[0]

    @staticmethod
    def _parse_ercot_timestamps(raw: pd.Series) -> pd.DatetimeIndex:
        """Parse ERCOT's ``Hour_Ending`` strings into a DatetimeIndex.

        Handles the special ``24:00`` hour that ERCOT uses for midnight
        of the next day.
        """
        # Some entries include DST flags like " DST" — strip them
        cleaned = raw.str.replace(r"\s*DST\s*$", "", regex=True)

        # Handle 24:00 → 00:00 + 1 day
        has_24 = cleaned.str.contains("24:00")

        # Replace 24:00 with 00:00 for parsing
        parseable = cleaned.str.replace("24:00", "00:00", regex=False)

        timestamps = pd.to_datetime(parseable, format="mixed", dayfirst=False)

        # Add one day where original was 24:00
        timestamps = timestamps.where(~has_24, timestamps + pd.Timedelta(days=1))

        return pd.DatetimeIndex(timestamps)
