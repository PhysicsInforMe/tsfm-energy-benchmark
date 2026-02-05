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

        Chronos-Bolt natively produces 9 quantiles (0.1, 0.2, ..., 0.9).
        We use the median (index 4, q0.5) as the point forecast, and
        synthesise approximate samples from a Gaussian fitted to the
        inter-quantile range (q0.1 / q0.9).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        context_tensor = torch.tensor(
            context.values, dtype=torch.float32
        ).unsqueeze(0)  # batch dim

        forecast = self.pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
        )
        # forecast shape: (batch, 9, prediction_length)
        # 9 quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        forecast = forecast[0]  # remove batch dim

        q10 = forecast[0].numpy()   # index 0 → q0.1
        point_forecast = forecast[4].numpy()  # index 4 → q0.5 (median)
        q90 = forecast[8].numpy()   # index 8 → q0.9

        # Approximate samples from the quantile spread
        std_approx = np.maximum((q90 - q10) / 2.56, 1e-6)
        samples = np.random.normal(
            loc=point_forecast,
            scale=std_approx,
            size=(num_samples, prediction_length),
        )

        return point_forecast, samples
