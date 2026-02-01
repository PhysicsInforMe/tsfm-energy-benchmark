"""Wrapper for Facebook Prophet.

Prophet handles multiple seasonalities (daily, weekly, yearly) natively
and is a strong baseline for energy load forecasting.

Requires: ``pip install prophet``
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .base import ForecastModel

logger = logging.getLogger(__name__)


class ProphetModel(ForecastModel):
    """Forecasting with Facebook Prophet.

    Args:
        yearly_seasonality: Enable yearly seasonality component.
        weekly_seasonality: Enable weekly seasonality component.
        daily_seasonality: Enable daily seasonality component.
        changepoint_prior_scale: Flexibility of trend changepoints.
        max_train_hours: Maximum training hours to use (Prophet can be
            slow on very long series).
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
        max_train_hours: int = 8760,  # 1 year
    ) -> None:
        super().__init__(name="Prophet", requires_gpu=False)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.max_train_hours = max_train_hours
        self._model = None

    def fit(self, train_data: pd.Series) -> ProphetModel:
        """Fit Prophet on the training series."""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "Prophet is required. Install with: pip install prophet"
            )

        # Prophet expects a DataFrame with columns 'ds' and 'y'
        df = pd.DataFrame({
            "ds": train_data.index,
            "y": train_data.values,
        })

        # Limit training data for speed
        if len(df) > self.max_train_hours:
            df = df.iloc[-self.max_train_hours:]

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        # Suppress Stan logging
        self._model.fit(df, suppress_logging=True)

        self._is_fitted = True
        logger.info("Prophet fitted on %d observations", len(df))
        return self

    def predict(
        self,
        context: pd.Series,
        prediction_length: int,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate forecast with uncertainty intervals.

        Prophet produces ``yhat``, ``yhat_lower``, ``yhat_upper``.
        We use these to build approximate Gaussian samples.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        # Build future dataframe starting from the end of context
        last_ts = context.index[-1]
        freq = pd.infer_freq(context.index) or "h"
        future_dates = pd.date_range(
            start=last_ts + pd.tseries.frequencies.to_offset(freq),
            periods=prediction_length,
            freq=freq,
        )
        future_df = pd.DataFrame({"ds": future_dates})

        forecast = self._model.predict(future_df)

        point_forecast = forecast["yhat"].values[:prediction_length]
        yhat_lower = forecast["yhat_lower"].values[:prediction_length]
        yhat_upper = forecast["yhat_upper"].values[:prediction_length]

        # Approximate samples from the 80% CI
        std_approx = np.maximum((yhat_upper - yhat_lower) / 2.56, 1e-6)
        samples = np.random.normal(
            loc=point_forecast,
            scale=std_approx,
            size=(num_samples, prediction_length),
        )

        return point_forecast, samples
