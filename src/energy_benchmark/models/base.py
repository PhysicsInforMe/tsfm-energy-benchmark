"""Abstract base class for all forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """Base class that every model wrapper must implement.

    Provides a uniform interface so the benchmark runner can treat
    all models identically.

    Args:
        name: Human-readable model identifier.
        requires_gpu: Whether the model needs a CUDA device.
    """

    def __init__(self, name: str, requires_gpu: bool = False) -> None:
        self.name = name
        self.requires_gpu = requires_gpu
        self._is_fitted = False

    @abstractmethod
    def fit(self, train_data: pd.Series) -> ForecastModel:
        """Train or initialise the model.

        For zero-shot foundation models this may simply load weights.

        Args:
            train_data: Training series with DatetimeIndex.

        Returns:
            ``self`` for method chaining.
        """

    @abstractmethod
    def predict(
        self,
        context: pd.Series,
        prediction_length: int,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate forecasts.

        Args:
            context: Recent history used as model input.
            prediction_length: Number of future steps to forecast.
            num_samples: Number of probabilistic samples (if supported).

        Returns:
            ``(point_forecast, samples)`` where *point_forecast* has shape
            ``(prediction_length,)`` and *samples* has shape
            ``(num_samples, prediction_length)`` or is ``None``.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
