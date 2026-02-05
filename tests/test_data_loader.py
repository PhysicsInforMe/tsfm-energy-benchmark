"""Tests for the ERCOT data loader (EIA API) and preprocessing utilities."""

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from energy_benchmark.data.ercot_loader import ERCOTLoader
from energy_benchmark.data.preprocessing import (
    preprocess_series,
    normalize,
    denormalize,
    create_splits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eia_response(year: int, n_rows: int = 100) -> dict:
    """Create a fake EIA API JSON response for testing."""
    idx = pd.date_range(f"{year}-01-01", periods=n_rows, freq="h")
    rows = [
        {
            "period": ts.strftime("%Y-%m-%dT%H"),
            "respondent": "ERCO",
            "respondent-name": "Electric Reliability Council of Texas, Inc.",
            "type": "D",
            "type-name": "Demand",
            "value": str(int(40000 + 5000 * np.sin(i * 2 * np.pi / 24))),
            "value-units": "megawatthours",
        }
        for i, ts in enumerate(idx)
    ]
    return {
        "response": {
            "total": n_rows,
            "data": rows,
        }
    }


# ---------------------------------------------------------------------------
# ERCOTLoader tests
# ---------------------------------------------------------------------------

class TestERCOTLoader:
    """Tests for ERCOTLoader initialisation and EIA API integration."""

    def test_default_years(self):
        loader = ERCOTLoader()
        assert loader.years == [2020, 2021, 2022, 2023, 2024]

    def test_custom_years(self):
        loader = ERCOTLoader(years=[2022, 2023])
        assert loader.years == [2022, 2023]

    def test_unsupported_year_raises(self):
        with pytest.raises(ValueError, match="not available before"):
            ERCOTLoader(years=[2010])

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"EIA_API_KEY": "MY_KEY_123"}):
            loader = ERCOTLoader()
            assert loader.api_key == "MY_KEY_123"

    def test_api_key_explicit(self):
        loader = ERCOTLoader(api_key="EXPLICIT_KEY")
        assert loader.api_key == "EXPLICIT_KEY"

    def test_api_key_default_demo(self):
        with patch.dict("os.environ", {}, clear=True):
            loader = ERCOTLoader()
            assert loader.api_key == "DEMO_KEY"

    def test_split_boundaries(self):
        idx = pd.date_range("2020-01-01", "2024-06-30", freq="h")
        series = pd.Series(np.random.randn(len(idx)), index=idx, name="load_mw")

        loader = ERCOTLoader()
        train, val, test = loader.split(series)

        assert train.index.max() <= pd.Timestamp("2022-12-31 23:00")
        assert val.index.min() > pd.Timestamp("2022-12-31")
        assert val.index.max() <= pd.Timestamp("2023-06-30 23:00")
        assert test.index.min() > pd.Timestamp("2023-06-30")

    def test_split_no_overlap(self):
        idx = pd.date_range("2020-01-01", "2024-06-30", freq="h")
        series = pd.Series(np.ones(len(idx)), index=idx, name="load_mw")

        loader = ERCOTLoader()
        train, val, test = loader.split(series)

        assert len(train) + len(val) + len(test) == len(series)

    def test_fetch_year_with_mock(self):
        """Test that _fetch_year correctly parses EIA API responses."""
        fake_response = _make_eia_response(2023, n_rows=48)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        loader = ERCOTLoader(years=[2023])

        with patch("energy_benchmark.data.ercot_loader.requests.get", return_value=mock_resp):
            df = loader._fetch_year(2023)

        assert len(df) == 48
        assert "ERCOT" in df.columns
        assert df.index.name == "timestamp"
        assert df["ERCOT"].dtype in [np.float64, np.int64]

    def test_fetch_year_pagination(self):
        """Test that pagination fetches all pages."""
        # First page: 50 rows out of 100 total
        page1 = _make_eia_response(2023, n_rows=50)
        page1["response"]["total"] = 100

        # Second page: next 50 rows
        idx2 = pd.date_range("2023-01-03 02:00", periods=50, freq="h")
        rows2 = [
            {
                "period": ts.strftime("%Y-%m-%dT%H"),
                "respondent": "ERCO",
                "type": "D",
                "value": str(40000 + i),
                "value-units": "megawatthours",
            }
            for i, ts in enumerate(idx2)
        ]
        page2 = {"response": {"total": 100, "data": rows2}}

        mock_resp1 = MagicMock()
        mock_resp1.json.return_value = page1
        mock_resp1.raise_for_status = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.json.return_value = page2
        mock_resp2.raise_for_status = MagicMock()

        loader = ERCOTLoader(years=[2023])

        with patch(
            "energy_benchmark.data.ercot_loader.requests.get",
            side_effect=[mock_resp1, mock_resp2],
        ):
            df = loader._fetch_year(2023)

        assert len(df) == 100

    def test_fetch_year_empty_raises(self):
        """Test that empty API response raises RuntimeError."""
        empty = {"response": {"total": 0, "data": []}}
        mock_resp = MagicMock()
        mock_resp.json.return_value = empty
        mock_resp.raise_for_status = MagicMock()

        loader = ERCOTLoader(years=[2023])

        with patch("energy_benchmark.data.ercot_loader.requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="No data returned"):
                loader._fetch_year(2023)

    def test_load_with_cache(self, tmp_path):
        """Test that load() uses parquet cache on second call."""
        # Create a fake cached parquet file
        idx = pd.date_range("2023-01-01", periods=100, freq="h")
        cached_df = pd.DataFrame(
            {"ERCOT": np.ones(100) * 40000},
            index=pd.DatetimeIndex(idx, name="timestamp"),
        )
        cache_file = tmp_path / "ercot_2023.parquet"
        cached_df.to_parquet(cache_file)

        loader = ERCOTLoader(years=[2023], data_dir=tmp_path)
        series = loader.load()

        assert len(series) == 100
        assert series.name == "load_mw"


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    """Tests for preprocessing utilities."""

    def _make_series(self, n: int = 100) -> pd.Series:
        idx = pd.date_range("2023-01-01", periods=n, freq="h")
        return pd.Series(np.random.randn(n) * 1000 + 40000, index=idx)

    def test_preprocess_no_nans(self):
        s = self._make_series()
        s.iloc[10] = np.nan
        result = preprocess_series(s)
        assert result.isna().sum() == 0

    def test_normalize_standard_roundtrip(self):
        s = self._make_series()
        normed, params = normalize(s, method="standard")
        recovered = denormalize(normed, params)
        np.testing.assert_allclose(recovered.values, s.values, atol=1e-6)

    def test_normalize_minmax_roundtrip(self):
        s = self._make_series()
        normed, params = normalize(s, method="minmax")
        recovered = denormalize(normed, params)
        np.testing.assert_allclose(recovered.values, s.values, atol=1e-6)

    def test_create_splits_sizes(self):
        idx = pd.date_range("2020-01-01", "2024-06-30", freq="h")
        s = pd.Series(np.ones(len(idx)), index=idx)
        train, val, test = create_splits(s, "2022-12-31", "2023-06-30")
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(s)
