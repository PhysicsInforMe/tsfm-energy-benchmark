"""
TinyTimeMixer (TTM) Model Wrapper.

TinyTimeMixer is a tiny pre-trained model for time series forecasting developed by IBM Research.
It achieves competitive performance with less than 1M parameters.

Reference: https://github.com/ibm-granite/granite-tsfm
Paper: "Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from .base import ForecastModel


class TinyTimeMixerModel(ForecastModel):
    """
    TinyTimeMixer (TTM) time series foundation model.

    Uses the granite-tsfm library to load pretrained TTM models from HuggingFace.

    Parameters
    ----------
    model_version : str
        Version of TTM model: 'r1' or 'r2' (r2 is recommended, trained on larger dataset)
    device : str
        Device to run inference on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        model_version: str = "r2",
        device: str = "cpu",
    ):
        self.model_version = model_version
        self.device = device
        self._model = None
        self._context_length = None
        self._default_prediction_length = None

    def fit(self, train_data: pd.Series) -> None:
        """Load the pretrained model (no training needed)."""
        self._load_model()

    def _load_model(self):
        """Load the TTM model from HuggingFace."""
        if self._model is not None:
            return  # Already loaded

        import torch
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        # Select model repo based on version
        if self.model_version == "r1":
            repo_id = "ibm-granite/granite-timeseries-ttm-r1"
        else:
            repo_id = "ibm-granite/granite-timeseries-ttm-r2"

        # Load model
        self._model = TinyTimeMixerForPrediction.from_pretrained(repo_id)

        # Store config values
        self._context_length = self._model.config.context_length
        self._default_prediction_length = self._model.config.prediction_length

        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()
        else:
            self._model = self._model.cpu()

        self._model.eval()

    def predict(
        self,
        context: pd.Series,
        prediction_length: int = 24,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate forecasts using TinyTimeMixer.

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
            Point forecast (mean of distribution)
        samples : np.ndarray
            Probabilistic samples of shape (num_samples, prediction_length)
        """
        self._load_model()

        import torch

        # Prepare input data
        ctx = context.values.astype(np.float32)

        # TTM expects input shape (batch, context_length, num_input_channels)
        # Pad or truncate context to match model's expected context length
        if len(ctx) > self._context_length:
            ctx = ctx[-self._context_length:]
        elif len(ctx) < self._context_length:
            # Pad with zeros at the beginning
            pad_length = self._context_length - len(ctx)
            ctx = np.concatenate([np.zeros(pad_length, dtype=np.float32), ctx])

        # Reshape for model: (batch=1, context_length, channels=1)
        input_tensor = torch.tensor(ctx).unsqueeze(0).unsqueeze(-1)

        if self.device == "cuda" and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        # Generate forecast
        with torch.no_grad():
            output = self._model(past_values=input_tensor)

        # Extract predictions
        # TTM outputs prediction_outputs of shape (batch, prediction_length, channels)
        if hasattr(output, 'prediction_outputs') and output.prediction_outputs is not None:
            pred = output.prediction_outputs
        else:
            raise RuntimeError("No predictions returned from model")

        # pred shape: (batch, prediction_length, channels)
        pred = pred.cpu().numpy()
        point_forecast = pred[0, :, 0]  # (prediction_length,)

        # Get scale and loc for proper denormalization if available
        if hasattr(output, 'loc') and output.loc is not None:
            loc = output.loc.cpu().numpy()[0, 0]
            scale = output.scale.cpu().numpy()[0, 0] if hasattr(output, 'scale') else 1.0
            # Point forecast is already denormalized by the model
        else:
            loc, scale = 0.0, 1.0

        # Truncate or extend to match requested prediction_length
        if len(point_forecast) > prediction_length:
            point_forecast = point_forecast[:prediction_length]
        elif len(point_forecast) < prediction_length:
            # Extend using last values (simple extrapolation)
            extension = np.full(
                prediction_length - len(point_forecast),
                point_forecast[-1]
            )
            point_forecast = np.concatenate([point_forecast, extension])

        # Generate samples from the distribution if available
        samples = None
        if hasattr(output, 'distribution') and output.distribution is not None:
            try:
                dist = output.distribution
                # Sample from the distribution
                samples_tensor = dist.sample((num_samples,))
                samples = samples_tensor.cpu().numpy()
                # samples shape: (num_samples, batch, prediction_length, channels)
                samples = samples[:, 0, :prediction_length, 0]
            except Exception:
                # If distribution sampling fails, generate samples using noise
                samples = self._generate_samples_from_point(
                    point_forecast, num_samples
                )
        else:
            # Generate samples using estimated uncertainty
            samples = self._generate_samples_from_point(point_forecast, num_samples)

        return point_forecast, samples

    def _generate_samples_from_point(
        self,
        point_forecast: np.ndarray,
        num_samples: int,
        noise_scale: float = 0.1,
    ) -> np.ndarray:
        """
        Generate samples from point forecast by adding scaled noise.

        Parameters
        ----------
        point_forecast : np.ndarray
            Point forecast of shape (prediction_length,)
        num_samples : int
            Number of samples to generate
        noise_scale : float
            Scale of noise relative to forecast magnitude

        Returns
        -------
        samples : np.ndarray
            Generated samples of shape (num_samples, prediction_length)
        """
        prediction_length = len(point_forecast)
        # Estimate noise scale from forecast magnitude
        scale = np.abs(point_forecast).mean() * noise_scale

        # Generate samples
        noise = np.random.randn(num_samples, prediction_length) * scale
        samples = point_forecast[np.newaxis, :] + noise

        return samples

    @property
    def name(self) -> str:
        """Return model name."""
        return f"TTM-{self.model_version}"
