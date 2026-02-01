"""Statistical baseline models: Seasonal Naive and ARIMA."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .base import ForecastModel

logger = logging.getLogger(__name__)


class SeasonalNaiveModel(ForecastModel):
    """Seasonal naive baseline: forecast = value from the same hour one cycle ago.

    Args:
        seasonality: Number of steps in one seasonal cycle.
            168 = weekly for hourly data, 24 = daily.
    """

    def __init__(self, seasonality: int = 168) -> None:
        super().__init__(name=f"SeasonalNaive-{seasonality}", requires_gpu=False)
        self.seasonality = seasonality

    def fit(self, train_data: pd.Series) -> SeasonalNaiveModel:
        self._is_fitted = True
        return self

    def predict(
        self,
        context: pd.Series,
        prediction_length: int,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        values = context.values
        forecast = np.empty(prediction_length)
        for i in range(prediction_length):
            # Look back by one seasonal cycle
            idx = len(values) - self.seasonality + (i % self.seasonality)
            if idx < 0:
                idx = i % len(values)
            forecast[i] = values[idx]

        return forecast, None


class ARIMAModel(ForecastModel):
    """SARIMA model via pmdarima.

    Args:
        order: ARIMA (p, d, q) order.
        seasonal_order: Seasonal (P, D, Q, m) order.
        auto: If True, use ``pmdarima.auto_arima`` to find best order.
        max_context: Maximum context length to use for fitting (to limit
            compute time).
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (2, 1, 2),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
        auto: bool = False,
        max_context: int = 2016,  # 12 weeks of hourly data
    ) -> None:
        name = "Auto-ARIMA" if auto else f"SARIMA{order}x{seasonal_order}"
        super().__init__(name=name, requires_gpu=False)
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto = auto
        self.max_context = max_context
        self._model = None

    def fit(self, train_data: pd.Series) -> ARIMAModel:
        self._is_fitted = True
        return self

    def predict(
        self,
        context: pd.Series,
        prediction_length: int,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError(
                "pmdarima is required for ARIMA models. "
                "Install with: pip install pmdarima"
            )

        # Limit context to keep fitting tractable
        ctx = context.values[-self.max_context :]

        if self.auto:
            model = pm.auto_arima(
                ctx,
                seasonal=True,
                m=self.seasonal_order[3],
                suppress_warnings=True,
                error_action="ignore",
                stepwise=True,
            )
        else:
            model = pm.ARIMA(
                order=self.order,
                seasonal_order=self.seasonal_order,
                suppress_warnings=True,
            )
            model.fit(ctx)

        # Point forecast
        point_forecast = model.predict(n_periods=prediction_length)

        # Confidence intervals → approximate samples
        _, conf_int = model.predict(
            n_periods=prediction_length, return_conf_int=True, alpha=0.2
        )
        lower, upper = conf_int[:, 0], conf_int[:, 1]
        std_approx = (upper - lower) / 2.56  # ~80% CI for normal
        samples = np.random.normal(
            loc=point_forecast,
            scale=np.maximum(std_approx, 1e-6),
            size=(num_samples, prediction_length),
        )

        return point_forecast, samples
