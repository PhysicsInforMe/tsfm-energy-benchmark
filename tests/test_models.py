"""Tests for model wrappers.

Foundation model tests use mocks to avoid downloading multi-GB checkpoints.
Statistical models are tested end-to-end on synthetic data.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from energy_benchmark.models.base import ForecastModel
from energy_benchmark.models.statistical import SeasonalNaiveModel, ARIMAModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(n: int = 512) -> pd.Series:
    """Synthetic hourly load series."""
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    np.random.seed(0)
    values = (
        40000
        + 5000 * np.sin(np.arange(n) * 2 * np.pi / 24)
        + np.random.randn(n) * 500
    )
    return pd.Series(values, index=idx, name="load_mw")


# ---------------------------------------------------------------------------
# Base class contract
# ---------------------------------------------------------------------------

class TestBaseContract:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ForecastModel(name="test")

    def test_repr(self):
        model = SeasonalNaiveModel(seasonality=24)
        assert "SeasonalNaive" in repr(model)


# ---------------------------------------------------------------------------
# Seasonal Naive
# ---------------------------------------------------------------------------

class TestSeasonalNaive:
    def test_predict_length(self):
        ctx = _make_context(512)
        model = SeasonalNaiveModel(seasonality=168)
        model.fit(ctx)
        point, samples = model.predict(ctx, prediction_length=48)
        assert point.shape == (48,)
        assert samples is None

    def test_predict_matches_lag(self):
        ctx = _make_context(200)
        model = SeasonalNaiveModel(seasonality=24)
        model.fit(ctx)
        point, _ = model.predict(ctx, prediction_length=24)
        expected = ctx.values[-24:]
        np.testing.assert_array_equal(point, expected)

    def test_raises_before_fit(self):
        model = SeasonalNaiveModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(_make_context(100), prediction_length=24)


# ---------------------------------------------------------------------------
# ARIMA (mocked pmdarima)
# ---------------------------------------------------------------------------

class TestARIMA:
    def test_predict_with_mock(self):
        pred_len = 24
        fake_forecast = np.ones(pred_len) * 40000
        fake_ci = np.column_stack([
            fake_forecast - 1000,
            fake_forecast + 1000,
        ])

        mock_arima_instance = MagicMock()
        mock_arima_instance.predict.side_effect = [
            fake_forecast,
            (fake_forecast, fake_ci),
        ]

        # Create a fake pmdarima module
        mock_pm = ModuleType("pmdarima")
        mock_pm.ARIMA = MagicMock(return_value=mock_arima_instance)

        with patch.dict(sys.modules, {"pmdarima": mock_pm}):
            ctx = _make_context(200)
            model = ARIMAModel(order=(1, 0, 0), seasonal_order=(0, 0, 0, 24))
            model.fit(ctx)
            point, samples = model.predict(ctx, prediction_length=pred_len)

        assert point.shape == (pred_len,)
        assert samples.shape[0] == 100
        assert samples.shape[1] == pred_len


# ---------------------------------------------------------------------------
# Chronos-Bolt (mocked pipeline)
# ---------------------------------------------------------------------------

class TestChronosBolt:
    def test_predict_with_mock(self):
        pred_len = 24

        mock_pipeline = MagicMock()
        fake_output = torch.randn(1, 3, pred_len)
        mock_pipeline.predict.return_value = fake_output

        mock_bolt_cls = MagicMock()
        mock_bolt_cls.from_pretrained.return_value = mock_pipeline

        # Fake the chronos module with ChronosBoltPipeline
        mock_chronos = ModuleType("chronos")
        mock_chronos.ChronosBoltPipeline = mock_bolt_cls

        with patch.dict(sys.modules, {"chronos": mock_chronos}):
            from energy_benchmark.models.chronos_bolt import ChronosBoltModel

            model = ChronosBoltModel(model_size="tiny", device="cpu")
            ctx = _make_context(256)
            model.fit(ctx)
            point, samples = model.predict(ctx, prediction_length=pred_len)

        assert point.shape == (pred_len,)
        assert samples.shape == (100, pred_len)

    def test_raises_before_fit(self):
        from energy_benchmark.models.chronos_bolt import ChronosBoltModel

        model = ChronosBoltModel(device="cpu")
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(_make_context(100), prediction_length=24)


# ---------------------------------------------------------------------------
# Chronos-2 (mocked pipeline)
# ---------------------------------------------------------------------------

class TestChronos2:
    def test_predict_with_mock(self):
        pred_len = 24
        num_samples = 50

        mock_pipeline = MagicMock()
        fake_samples = torch.randn(1, num_samples, pred_len)
        mock_pipeline.predict.return_value = fake_samples

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

        mock_chronos = ModuleType("chronos")
        mock_chronos.ChronosPipeline = mock_pipeline_cls

        with patch.dict(sys.modules, {"chronos": mock_chronos}):
            from energy_benchmark.models.chronos2 import Chronos2Model

            model = Chronos2Model(device="cpu")
            ctx = _make_context(256)
            model.fit(ctx)
            point, samples = model.predict(
                ctx, prediction_length=pred_len, num_samples=num_samples
            )

        assert point.shape == (pred_len,)
        assert samples.shape == (num_samples, pred_len)


# ---------------------------------------------------------------------------
# Prophet (mocked)
# ---------------------------------------------------------------------------

class TestProphet:
    def test_predict_with_mock(self):
        pred_len = 24

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.DataFrame({
            "yhat": np.ones(pred_len) * 40000,
            "yhat_lower": np.ones(pred_len) * 39000,
            "yhat_upper": np.ones(pred_len) * 41000,
        })

        mock_prophet_cls = MagicMock(return_value=mock_model)

        mock_prophet_mod = ModuleType("prophet")
        mock_prophet_mod.Prophet = mock_prophet_cls

        with patch.dict(sys.modules, {"prophet": mock_prophet_mod}):
            from energy_benchmark.models.prophet_model import ProphetModel

            model = ProphetModel(max_train_hours=200)
            ctx = _make_context(200)
            model.fit(ctx)
            point, samples = model.predict(ctx, prediction_length=pred_len)

        assert point.shape == (pred_len,)
        assert samples.shape == (100, pred_len)
        np.testing.assert_allclose(point, 40000, atol=1)

    def test_raises_before_fit(self):
        from energy_benchmark.models.prophet_model import ProphetModel

        model = ProphetModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(_make_context(100), prediction_length=24)
