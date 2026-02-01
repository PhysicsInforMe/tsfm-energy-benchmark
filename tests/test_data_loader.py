"""Tests for the ERCOT data loader."""

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


class TestERCOTLoader:
    """Tests for ERCOTLoader initialisation and validation."""

    def test_default_years(self):
        loader = ERCOTLoader()
        assert loader.years == [2020, 2021, 2022, 2023, 2024]

    def test_custom_years(self):
        loader = ERCOTLoader(years=[2022, 2023])
        assert loader.years == [2022, 2023]

    def test_unsupported_year_raises(self):
        with pytest.raises(ValueError, match="No known URL"):
            ERCOTLoader(years=[2015])

    def test_split_boundaries(self):
        # Create synthetic hourly data 2020-2024
        idx = pd.date_range("2020-01-01", "2024-06-30", freq="h")
        series = pd.Series(np.random.randn(len(idx)), index=idx, name="load_mw")

        loader = ERCOTLoader()
        train, val, test = loader.split(series)

        assert train.index.max() <= pd.Timestamp("2022-12-31 23:00")
        assert val.index.min() > pd.Timestamp("2022-12-31")
        assert val.index.max() <= pd.Timestamp("2023-06-30 23:00")
        assert test.index.min() > pd.Timestamp("2023-06-30")

    def test_parse_ercot_timestamps_handles_24(self):
        raw = pd.Series(["01/01/2023 01:00", "01/01/2023 24:00"])
        result = ERCOTLoader._parse_ercot_timestamps(raw)

        assert result[0] == pd.Timestamp("2023-01-01 01:00")
        assert result[1] == pd.Timestamp("2023-01-02 00:00")

    def test_parse_ercot_timestamps_strips_dst(self):
        raw = pd.Series(["03/10/2023 03:00 DST"])
        result = ERCOTLoader._parse_ercot_timestamps(raw)
        assert result[0] == pd.Timestamp("2023-03-10 03:00")


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
