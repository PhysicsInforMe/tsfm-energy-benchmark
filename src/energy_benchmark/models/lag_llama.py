"""Wrapper for the Lag-Llama foundation model.

Lag-Llama is a decoder-only transformer that uses lag features for
probabilistic time-series forecasting.  It was the first open-source TSFM.

Requires:
    - Clone of https://github.com/time-series-foundation-models/lag-llama
    - ``pip install gluonts[torch]<=0.14.4 pytorch-lightning``
    - ``huggingface-cli download time-series-foundation-models/Lag-Llama
       lag-llama.ckpt --local-dir ./lag-llama``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .base import ForecastModel

logger = logging.getLogger(__name__)


def _patch_torch_load():
    """Patch torch.load to use weights_only=False for PyTorch 2.6+ compatibility."""
    import torch
    if hasattr(torch, '_original_load'):
        return  # Already patched
    torch._original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return torch._original_load(*args, **kwargs)
    torch.load = _patched_load


class LagLlamaModel(ForecastModel):
    """Zero-shot forecasting with Lag-Llama.

    Args:
        ckpt_path: Path to the ``lag-llama.ckpt`` checkpoint file.
        context_length: Number of past time-steps to feed the model.
        device: ``'cuda'`` or ``'cpu'``.
    """

    def __init__(
        self,
        ckpt_path: str | Path = "lag-llama/lag-llama.ckpt",
        context_length: int = 512,
        device: str = "cuda",
    ) -> None:
        super().__init__(name="Lag-Llama", requires_gpu=(device == "cuda"))
        self.ckpt_path = Path(ckpt_path)
        self.context_length = context_length
        self.device = device
        self._predictor = None
        self._prediction_length = None

    def fit(self, train_data: pd.Series) -> "LagLlamaModel":
        """Load checkpoint - Lag-Llama is zero-shot so no actual fitting."""
        # Patch torch.load for PyTorch 2.6+ compatibility
        _patch_torch_load()

        import torch

        # Lag-Llama repo must be on sys.path
        import sys
        repo_dir = str(self.ckpt_path.parent)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self._device = device
        self._is_fitted = True

        logger.info("Lag-Llama initialized from %s", self.ckpt_path)
        return self

    def _get_predictor(self, prediction_length: int):
        """Create predictor with specified prediction length."""
        import torch
        from lag_llama.gluon.estimator import LagLlamaEstimator

        ckpt = torch.load(self.ckpt_path, map_location=self._device, weights_only=False)
        estimator_args = ckpt.get("hyper_parameters", {}).get("model_kwargs", {})

        estimator = LagLlamaEstimator(
            ckpt_path=str(self.ckpt_path),
            prediction_length=prediction_length,
            context_length=self.context_length,
            input_size=estimator_args.get("input_size", 1),
            n_layer=estimator_args.get("n_layer", 8),
            n_embd_per_head=estimator_args.get("n_embd_per_head", 32),
            n_head=estimator_args.get("n_head", 4),
            scaling=estimator_args.get("scaling", "mean"),
            time_feat=estimator_args.get("time_feat", False),
            batch_size=1,
            num_parallel_samples=100,
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)

        return predictor

    def predict(
        self,
        context: pd.Series,
        prediction_length: int,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Generate probabilistic forecast using GluonTS sampling."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")

        from gluonts.dataset.pandas import PandasDataset

        # Get or create predictor for this prediction length
        if self._predictor is None or self._prediction_length != prediction_length:
            self._predictor = self._get_predictor(prediction_length)
            self._prediction_length = prediction_length

        # Build GluonTS dataset from context
        ctx = context.iloc[-self.context_length:].copy()
        ctx.index = pd.to_datetime(ctx.index)

        # Create dataset in the format GluonTS expects
        # Ensure float32 for model compatibility
        ds = PandasDataset.from_long_dataframe(
            pd.DataFrame({
                "target": ctx.values.astype(np.float32),
                "timestamp": ctx.index,
                "item_id": "series",
            }),
            target="target",
            timestamp="timestamp",
            item_id="item_id",
        )

        forecasts = list(self._predictor.predict(ds, num_samples=num_samples))
        if not forecasts:
            raise RuntimeError("Lag-Llama returned no forecasts")

        fc = forecasts[0]
        samples = fc.samples  # (num_samples, prediction_length)
        point_forecast = np.median(samples, axis=0)

        return point_forecast, samples
