"""ERCOT hourly load data loader via EIA Open Data API.

Retrieves historical hourly demand data for ERCOT (Electric Reliability Council
of Texas) from the U.S. Energy Information Administration (EIA) Open Data API v2.

The EIA API is free and publicly accessible.  A ``DEMO_KEY`` works out of the
box (rate-limited to ~30 req/hour).  For heavier usage, obtain a free key at
https://www.eia.gov/opendata/register.php and set the ``EIA_API_KEY``
environment variable.

API docs: https://www.eia.gov/opendata/documentation.php
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# EIA API v2 endpoint for Regional Electricity Data (hourly demand)
EIA_API_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

# Maximum rows the API returns per request
EIA_PAGE_SIZE = 5000

# EIA RTO hourly data is available from ~2015 onwards
MIN_YEAR = 2015

# Default target column name (for compatibility with rest of codebase)
TARGET_COLUMN = "ERCOT"


class ERCOTLoader:
    """Load ERCOT hourly demand data via the EIA Open Data API.

    Args:
        years: List of years to include.  Defaults to 2020-2024.
        data_dir: Directory for caching downloaded data as parquet files.
        target_column: Name for the target column in the output DataFrame.
        api_key: EIA API key.  Falls back to env var ``EIA_API_KEY``,
                 then to ``DEMO_KEY`` (rate-limited).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        years: Optional[List[int]] = None,
        data_dir: str | Path = "data/raw",
        target_column: str = TARGET_COLUMN,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.years = years or list(range(2020, 2025))
        self.data_dir = Path(data_dir)
        self.target_column = target_column
        self.api_key = api_key or os.environ.get("EIA_API_KEY", "DEMO_KEY")
        self.timeout = timeout

        unsupported = [y for y in self.years if y < MIN_YEAR]
        if unsupported:
            raise ValueError(
                f"EIA RTO data is not available before {MIN_YEAR}. "
                f"Unsupported year(s): {sorted(unsupported)}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, force_download: bool = False) -> pd.Series:
        """Fetch (if not cached) and return the full hourly load series.

        Args:
            force_download: Re-fetch from the API even if a local parquet
                            cache already exists.

        Returns:
            A pandas Series with DatetimeIndex (hourly frequency) containing
            system demand in MW for the requested years, sorted chronologically.
        """
        frames: list[pd.DataFrame] = []
        for year in sorted(self.years):
            df = self._load_year(year, force_download=force_download)
            frames.append(df)

        combined = pd.concat(frames, axis=0)
        combined = combined.sort_index()

        # Remove duplicate timestamps (rare, but possible at year boundaries)
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
        """Load a single year, fetching from the API if not cached."""
        cache_path = self.data_dir / f"ercot_{year}.parquet"

        if cache_path.exists() and not force_download:
            logger.debug("Loading cached data for %d", year)
            return pd.read_parquet(cache_path)

        # Fetch from API
        df = self._fetch_year(year)

        # Cache as parquet for fast subsequent loads
        self.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cached %d data to %s", year, cache_path)

        return df

    def _fetch_year(self, year: int) -> pd.DataFrame:
        """Fetch one year of hourly demand data from the EIA API.

        Handles pagination automatically (the API returns at most 5 000
        rows per request; a full year has ~8 760 rows).
        """
        start = f"{year}-01-01T00"
        end = f"{year}-12-31T23"

        all_rows: list[dict] = []
        offset = 0

        while True:
            params = {
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": "ERCO",
                "facets[type][]": "D",
                "start": start,
                "end": end,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": EIA_PAGE_SIZE,
                "api_key": self.api_key,
            }

            logger.info(
                "Fetching ERCOT %d data from EIA API (offset=%d)", year, offset
            )

            response = requests.get(
                EIA_API_URL, params=params, timeout=self.timeout
            )
            response.raise_for_status()

            payload = response.json()
            resp_body = payload.get("response", {})
            data = resp_body.get("data", [])
            total = resp_body.get("total", 0)

            if not data:
                break

            all_rows.extend(data)
            offset += len(data)

            if offset >= total:
                break

        if not all_rows:
            raise RuntimeError(
                f"No data returned from EIA API for ERCOT year {year}. "
                f"Check your API key and network connection."
            )

        logger.info("Fetched %d rows for %d", len(all_rows), year)

        # Build DataFrame
        df = pd.DataFrame(all_rows)
        timestamps = pd.to_datetime(df["period"])
        values = pd.to_numeric(df["value"], errors="coerce")

        result = pd.DataFrame(
            {self.target_column: values.values},
            index=pd.DatetimeIndex(timestamps, name="timestamp"),
        )
        result = result.dropna(subset=[self.target_column])
        result = result.sort_index()

        return result
