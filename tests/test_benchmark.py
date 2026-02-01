"""Tests for BenchmarkRunner and visualization."""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from energy_benchmark.models.statistical import SeasonalNaiveModel
from energy_benchmark.evaluation.benchmark import BenchmarkRunner, BenchmarkResults
from energy_benchmark.visualization.plots import (
    plot_comparison,
    plot_forecasts,
    plot_probabilistic_forecast,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_series(n: int = 3000) -> pd.Series:
    """Synthetic hourly series with daily seasonality."""
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    np.random.seed(42)
    values = (
        40000
        + 5000 * np.sin(np.arange(n) * 2 * np.pi / 24)
        + np.random.randn(n) * 300
    )
    return pd.Series(values, index=idx, name="load_mw")


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    def test_run_single_model_single_horizon(self):
        series = _make_series(3000)
        train = series.iloc[:2000]
        test = series.iloc[2000:]

        model = SeasonalNaiveModel(seasonality=24)
        model.fit(train)

        runner = BenchmarkRunner(
            models=[model],
            prediction_horizons=[24],
            context_lengths=[168],
            num_samples=10,
            metric_names=["mae", "rmse", "mase"],
        )
        results = runner.run(
            train,
            test,
            rolling_config={"step_size": 24, "num_windows": 3},
        )

        df = results.to_dataframe()
        assert len(df) == 1  # 1 model x 1 horizon x 1 ctx
        assert "mae" in df.columns
        assert "rmse" in df.columns
        assert "mase" in df.columns
        assert df["mae"].iloc[0] > 0
        assert len(results.forecasts) == 3  # 3 windows

    def test_run_multiple_horizons(self):
        series = _make_series(5000)
        train = series.iloc[:3000]
        test = series.iloc[3000:]

        model = SeasonalNaiveModel(seasonality=24)
        model.fit(train)

        runner = BenchmarkRunner(
            models=[model],
            prediction_horizons=[24, 168],
            context_lengths=[168],
            num_samples=10,
            metric_names=["mae"],
        )
        results = runner.run(
            train,
            test,
            rolling_config={"step_size": 168, "num_windows": 2},
        )

        df = results.to_dataframe()
        assert len(df) == 2  # 1 model x 2 horizons

    def test_results_to_dataframe(self):
        r = BenchmarkResults()
        r.records.append({"model": "A", "horizon": 24, "mae": 100.0})
        df = r.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Visualization (smoke tests — just check they don't crash)
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_plot_comparison(self):
        df = pd.DataFrame({
            "model": ["A", "A", "B", "B"],
            "horizon": [24, 168, 24, 168],
            "mae": [100, 200, 90, 180],
        })
        fig = plot_comparison(df, metric="mae")
        assert fig is not None

    def test_plot_forecasts(self):
        idx = pd.date_range("2023-06-01", periods=168, freq="h")
        actual = pd.Series(np.random.randn(168) * 1000 + 40000, index=idx)
        preds = {"ModelA": np.random.randn(168) * 1000 + 40000}
        fig = plot_forecasts(actual, preds)
        assert fig is not None

    def test_plot_probabilistic(self):
        n = 48
        actual = np.random.randn(n) * 1000 + 40000
        point = actual + np.random.randn(n) * 100
        samples = np.random.randn(50, n) * 500 + point[None, :]
        fig = plot_probabilistic_forecast(actual, point, samples)
        assert fig is not None
