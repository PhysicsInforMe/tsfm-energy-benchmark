"""
Moirai Time Series Foundation Model Wrapper.

Moirai is a universal time series forecasting transformer developed by Salesforce Research.
This wrapper supports Moirai 1.0, 1.1, and 2.0 variants.

Reference: https://github.com/SalesforceAIResearch/uni2ts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from .base import ForecastModel


class MoiraiModel(ForecastModel):
    """
    Moirai time series foundation model.

    Uses the uni2ts library to load pretrained Moirai models from HuggingFace.
    Supports Moirai 1.0, 1.1 (moirai), Moirai-MoE (moirai-moe), and Moirai 2.0 (moirai2).

    Parameters
    ----------
    model_type : str
        Type of Moirai model: 'moirai', 'moirai-moe', or 'moirai2'
    size : str
        Model size: 'small', 'base', or 'large'
    device : str
        Device to run inference on ('cpu' or 'cuda')
    patch_size : str or int
        Patch size for the model. 'auto' or one of 8, 16, 32, 64, 128
    """

    def __init__(
        self,
        model_type: str = "moirai2",
        size: str = "small",
        device: str = "cpu",
        patch_size: str | int = "auto",
    ):
        self.model_type = model_type
        self.size = size
        self.device = device
        self.patch_size = patch_size
        self._module = None
        self._forecast_cls = None
        self._module_cls = None

    def fit(self, train_data: pd.Series) -> None:
        """Load the pretrained model (no training needed)."""
        self._load_model()

    def _load_model(self):
        """Load the appropriate Moirai model variant."""
        if self._module is not None:
            return  # Already loaded

        import torch

        if self.model_type == "moirai2":
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
            self._forecast_cls = Moirai2Forecast
            self._module_cls = Moirai2Module
            repo_id = f"Salesforce/moirai-2.0-R-{self.size}"
        elif self.model_type == "moirai-moe":
            from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
            self._forecast_cls = MoiraiMoEForecast
            self._module_cls = MoiraiMoEModule
            repo_id = f"Salesforce/moirai-moe-1.0-R-{self.size}"
        else:  # moirai (1.0 or 1.1)
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            self._forecast_cls = MoiraiForecast
            self._module_cls = MoiraiModule
            repo_id = f"Salesforce/moirai-1.1-R-{self.size}"

        # Load pretrained module
        self._module = self._module_cls.from_pretrained(repo_id)

        # Move to appropriate device
        if self.device == "cpu":
            self._module = self._module.to(torch.float32)

    def predict(
        self,
        context: pd.Series,
        prediction_length: int = 24,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate forecasts using Moirai.

        Parameters
        ----------
        context : pd.Series
            Historical time series data
        prediction_length : int
            Number of time steps to forecast
        num_samples : int
            Number of probabilistic samples to generate

        Returns
        -------
        point_forecast : np.ndarray
            Point forecast (median of samples)
        samples : np.ndarray
            Probabilistic samples of shape (num_samples, prediction_length)
        """
        self._load_model()

        import torch
        from gluonts.dataset.pandas import PandasDataset
        from gluonts.dataset.split import split

        # Prepare context data
        ctx = context.copy()
        if not isinstance(ctx.index, pd.DatetimeIndex):
            ctx.index = pd.to_datetime(ctx.index)
        ctx = ctx.asfreq('h')

        # Convert to float32 for model compatibility
        values = ctx.values.astype(np.float32)

        # Create dataset for GluonTS
        ds = PandasDataset.from_long_dataframe(
            pd.DataFrame({
                "target": values,
                "timestamp": ctx.index,
                "item_id": "series",
            }),
            target="target",
            timestamp="timestamp",
            item_id="item_id",
            freq="h",
        )

        # Create forecast model with current parameters
        context_length = len(ctx)

        # Build kwargs for forecast class
        forecast_kwargs = {
            "module": self._module,
            "prediction_length": prediction_length,
            "context_length": context_length,
            "target_dim": 1,
            "feat_dynamic_real_dim": 0,
            "past_feat_dynamic_real_dim": 0,
        }

        # Add patch_size for non-moirai2 models
        if self.model_type != "moirai2":
            forecast_kwargs["patch_size"] = self.patch_size
            forecast_kwargs["num_samples"] = num_samples

        model = self._forecast_cls(**forecast_kwargs)

        # Create predictor and generate forecasts
        predictor = model.create_predictor(batch_size=1)

        # Generate forecast
        forecasts = list(predictor.predict(ds))

        if not forecasts:
            raise RuntimeError("No forecasts generated")

        forecast = forecasts[0]

        # Extract samples and point forecast
        if hasattr(forecast, 'samples') and forecast.samples is not None:
            samples = forecast.samples  # Shape: (num_samples, prediction_length)
            if samples.ndim == 3:
                samples = samples.squeeze(-1)  # Remove target dim if present
            point_forecast = np.median(samples, axis=0)
        elif hasattr(forecast, 'forecast_array'):
            # QuantileForecast - generate pseudo-samples from quantiles
            point_forecast = forecast.quantile(0.5)

            # Generate samples by sampling from quantile distribution
            # Use inverse CDF sampling with interpolation between quantiles
            quantile_levels = np.array([float(k) for k in forecast.forecast_keys])
            quantile_values = forecast.forecast_array  # Shape: (num_quantiles, horizon)

            samples = self._generate_samples_from_quantiles(
                quantile_levels, quantile_values, num_samples
            )
        else:
            # Fallback
            point_forecast = forecast.mean if hasattr(forecast, 'mean') else forecast.median
            samples = None

        return point_forecast, samples

    def _generate_samples_from_quantiles(
        self,
        quantile_levels: np.ndarray,
        quantile_values: np.ndarray,
        num_samples: int,
    ) -> np.ndarray:
        """
        Generate samples from quantile forecasts using inverse CDF interpolation.

        Parameters
        ----------
        quantile_levels : np.ndarray
            Quantile levels (e.g., [0.1, 0.2, ..., 0.9])
        quantile_values : np.ndarray
            Quantile values of shape (num_quantiles, horizon)
        num_samples : int
            Number of samples to generate

        Returns
        -------
        samples : np.ndarray
            Generated samples of shape (num_samples, horizon)
        """
        from scipy import interpolate

        horizon = quantile_values.shape[1]
        samples = np.zeros((num_samples, horizon))

        # Add boundary quantiles for better extrapolation
        extended_levels = np.concatenate([[0.01], quantile_levels, [0.99]])

        for t in range(horizon):
            q_vals = quantile_values[:, t]

            # Extrapolate to boundary quantiles using linear extension
            lower_ext = q_vals[0] - (q_vals[1] - q_vals[0]) * (quantile_levels[0] - 0.01) / (quantile_levels[1] - quantile_levels[0])
            upper_ext = q_vals[-1] + (q_vals[-1] - q_vals[-2]) * (0.99 - quantile_levels[-1]) / (quantile_levels[-1] - quantile_levels[-2])
            extended_vals = np.concatenate([[lower_ext], q_vals, [upper_ext]])

            # Create interpolation function (inverse CDF)
            inv_cdf = interpolate.interp1d(
                extended_levels, extended_vals,
                kind='linear', fill_value='extrapolate'
            )

            # Sample uniform random values and transform through inverse CDF
            u = np.random.uniform(0.05, 0.95, num_samples)  # Stay within reasonable range
            samples[:, t] = inv_cdf(u)

        return samples

    @property
    def name(self) -> str:
        """Return model name."""
        return f"Moirai-{self.model_type}-{self.size}"
