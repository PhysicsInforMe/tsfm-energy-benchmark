"""Wrapper for the Chronos-2 foundation model.

Chronos-2 is a second-generation time series foundation model released by
Amazon in 2025.  It uses an encoder-only architecture with group attention
and supports multivariate inputs and covariates.  Here we use it in
univariate zero-shot mode for consistency with the benchmark.

Unlike the original Chronos (T5-based, sample output) and Chronos-Bolt
(quantile output), Chronos-2 produces a fine-grained set of 21 quantiles.

Requires: ``pip install chronos-forecasting>=2.2``
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
        from chronos import Chronos2Pipeline  # lazy import

        model_id = "amazon/chronos-2"
        logger.info("Loading %s on %s", model_id, self.device)

        self.pipeline = Chronos2Pipeline.from_pretrained(
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

        Chronos-2 produces 21 quantiles (0.025, 0.05, 0.1, ..., 0.975).
        We use the median as the point forecast and synthesise approximate
        samples from the inter-quantile range.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        # Chronos-2 expects 3D input: (batch, n_variates, history_length)
        context_tensor = torch.tensor(
            context.values, dtype=torch.float32
        ).reshape(1, 1, -1)

        result_list = self.pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
        )
        # result_list is a list of tensors, one per series in the batch.
        # Each tensor has shape (n_variates, n_quantiles, prediction_length).
        forecast = result_list[0]  # first (only) series
        # For univariate: shape (1, 21, prediction_length) → squeeze variates
        forecast = forecast[0]  # shape: (21, prediction_length)

        n_quantiles = forecast.shape[0]
        mid = n_quantiles // 2  # median index (10 for 21 quantiles)

        point_forecast = forecast[mid].numpy()

        # Use q_low and q_high for sample approximation
        q_low = forecast[0].numpy()      # lowest quantile
        q_high = forecast[-1].numpy()     # highest quantile
        std_approx = np.maximum((q_high - q_low) / 3.92, 1e-6)  # ~95% CI
        samples = np.random.normal(
            loc=point_forecast,
            scale=std_approx,
            size=(num_samples, prediction_length),
        )

        return point_forecast, samples
