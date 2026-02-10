"""Wrapper for Facebook Prophet.

Prophet handles multiple seasonalities (daily, weekly, yearly) natively
and is a strong baseline for energy load forecasting.

Requires: ``pip install prophet``
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Suppress Prophet's verbose logging
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class ProphetModel:
    """Forecasting with Facebook Prophet.

    Prophet is a trained model (not zero-shot) - it fits on the context
    data at each prediction. This serves as a strong industry baseline.

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
        yearly_seasonality: bool = False,  # Usually not enough context
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
        max_train_hours: int = 2048,
        **kwargs  # Accept extra args for compatibility
    ) -> None:
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.max_train_hours = max_train_hours
        self._model = None

    def predict(
        self,
        context: pd.Series,
        prediction_length: int = 24,
        num_samples: int = 100,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit Prophet on context and generate forecasts.

        This method fits Prophet on the context data and then predicts.
        Prophet is NOT a zero-shot model - it trains on each context.

        Args:
            context: Historical values (pd.Series with DatetimeIndex or array)
            prediction_length: Number of steps to forecast
            num_samples: Number of samples for probabilistic forecast

        Returns:
            Tuple of (point_forecast, samples)
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "Prophet is required. Install with: pip install prophet"
            )

        # Handle both Series and array inputs
        if isinstance(context, pd.Series):
            timestamps = context.index
            values = context.values
        else:
            # Create synthetic timestamps
            context = np.asarray(context).flatten()
            n = len(context)
            end_time = pd.Timestamp.now().floor('H')
            timestamps = pd.date_range(end=end_time, periods=n, freq='H')
            values = context

        # Prophet expects a DataFrame with columns 'ds' and 'y'
        df = pd.DataFrame({
            "ds": timestamps,
            "y": values,
        })

        # Limit training data for speed
        if len(df) > self.max_train_hours:
            df = df.iloc[-self.max_train_hours:]

        # Fit Prophet (suppress all logging)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self._model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
            )

            # Fit with logging suppressed
            import logging as _logging
            _logging.getLogger('cmdstanpy').disabled = True
            self._model.fit(df)
            _logging.getLogger('cmdstanpy').disabled = False

        # Build future dataframe
        last_ts = df["ds"].iloc[-1]
        freq = pd.infer_freq(df["ds"]) or "h"
        future_dates = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1),
            periods=prediction_length,
            freq=freq,
        )
        future_df = pd.DataFrame({"ds": future_dates})

        # Generate forecast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._model.predict(future_df)

        point_forecast = forecast["yhat"].values[:prediction_length]
        yhat_lower = forecast["yhat_lower"].values[:prediction_length]
        yhat_upper = forecast["yhat_upper"].values[:prediction_length]

        # Approximate samples from the 80% CI (Prophet default)
        # 80% CI corresponds to ±1.28 std
        std_approx = np.maximum((yhat_upper - yhat_lower) / 2.56, 1e-6)
        samples = np.random.normal(
            loc=point_forecast,
            scale=std_approx,
            size=(num_samples, prediction_length),
        )

        return point_forecast, samples

    def __repr__(self) -> str:
        return f"ProphetModel(weekly={self.weekly_seasonality}, daily={self.daily_seasonality})"
