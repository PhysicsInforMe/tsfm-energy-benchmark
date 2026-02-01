"""Wrapper for the Chronos-2 foundation model.

Chronos-2 is an encoder-only model with group attention, released by Amazon
in late 2024.  It supports covariates and multivariate inputs but here we
use it in univariate zero-shot mode for consistency with the benchmark.

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


class Chronos2Model(ForecastModel):
    """Zero-shot forecasting with Chronos-2.

    Args:
        device: ``'cuda'`` or ``'cpu'``.
    """

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(name="Chronos-2", requires_gpu=(device == "cuda"))
        self.device = device
        self.pipeline = None

    def fit(self, train_data: pd.Series) -> Chronos2Model:
        """Load the pre-trained model."""
        from chronos import ChronosPipeline  # lazy import

        model_id = "amazon/chronos-t5-base"
        logger.info("Loading %s on %s", model_id, self.device)

        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        self._is_fitted = True
        return self

    def predict(
        self,
        context: pd.Series,
        prediction_length: int,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate sample-based probabilistic forecast.

        Unlike Chronos-Bolt, the original Chronos / Chronos-2 pipeline
        produces Monte-Carlo samples rather than quantiles.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        context_tensor = torch.tensor(
            context.values, dtype=torch.float32
        ).unsqueeze(0)

        forecast_samples = self.pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )
        # shape: (batch, num_samples, prediction_length)
        samples = forecast_samples[0].numpy()  # remove batch dim

        point_forecast = np.median(samples, axis=0)
        return point_forecast, samples
