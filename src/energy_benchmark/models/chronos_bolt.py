"""Wrapper for the Chronos-Bolt foundation model.

Chronos-Bolt uses a T5 encoder-decoder architecture with patch-based input.
It is ~250x faster than the original Chronos and produces probabilistic
forecasts directly as quantiles.

Requires: ``pip install chronos-forecasting``
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .base import ForecastModel

logger = logging.getLogger(__name__)


class ChronosBoltModel(ForecastModel):
    """Zero-shot forecasting with Chronos-Bolt.

    Args:
        model_size: One of ``'tiny'``, ``'mini'``, ``'small'``, ``'base'``.
        device: ``'cuda'`` or ``'cpu'``.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
    ) -> None:
        super().__init__(
            name=f"Chronos-Bolt-{model_size.capitalize()}",
            requires_gpu=(device == "cuda"),
        )
        self.model_size = model_size
        self.device = device
        self.pipeline = None

    def fit(self, train_data: pd.Series) -> ChronosBoltModel:
        """Load the pre-trained model (zero-shot — no training needed)."""
        from chronos import ChronosBoltPipeline  # lazy import

        model_id = f"amazon/chronos-bolt-{self.model_size}"
        logger.info("Loading %s on %s", model_id, self.device)

        self.pipeline = ChronosBoltPipeline.from_pretrained(
            model_id,
            device_map=self.device,
        )
        self._is_fitted = True
        return self

    def predict(
        self,
        context: pd.Series,
        prediction_length: int,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate probabilistic forecast via quantile output.

        Chronos-Bolt natively produces quantile forecasts.  We request the
        10th, 50th and 90th percentiles, use the median as the point
        forecast, and synthesise approximate samples from a Gaussian
        fitted to the inter-quantile range.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        context_tensor = torch.tensor(
            context.values, dtype=torch.float32
        ).unsqueeze(0)  # batch dim

        quantile_levels = [0.1, 0.5, 0.9]
        forecast = self.pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )
        # forecast shape: (batch, num_quantiles, prediction_length)
        forecast = forecast[0]  # remove batch dim

        point_forecast = forecast[1].numpy()  # median (q0.5)

        # Approximate samples from the quantile spread
        q10 = forecast[0].numpy()
        q90 = forecast[2].numpy()
        std_approx = np.maximum((q90 - q10) / 2.56, 1e-6)
        samples = np.random.normal(
            loc=point_forecast,
            scale=std_approx,
            size=(num_samples, prediction_length),
        )

        return point_forecast, samples
