"""Wrapper for the Lag-Llama foundation model.

Lag-Llama is a decoder-only transformer that uses lag features for
probabilistic time-series forecasting.  It was the first open-source TSFM.

Requires:
    - Clone of https://github.com/time-series-foundation-models/lag-llama
    - ``pip install -r lag-llama/requirements.txt``
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

    def fit(self, train_data: pd.Series) -> LagLlamaModel:
        """Load checkpoint and build the GluonTS predictor."""
        import torch

        # Lag-Llama repo must be on sys.path
        import sys
        repo_dir = str(self.ckpt_path.parent)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        from lag_llama.gluon.estimator import LagLlamaEstimator

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(self.ckpt_path, map_location=device)
        estimator_args = ckpt.get("hyper_parameters", {}).get("model_kwargs", {})

        estimator = LagLlamaEstimator(
            ckpt_path=str(self.ckpt_path),
            prediction_length=24,  # placeholder, overridden in predict
            context_length=self.context_length,
            input_size=estimator_args.get("input_size", 1),
            n_layer=estimator_args.get("n_layer", 8),
            n_embd_per_head=estimator_args.get("n_embd_per_head", 32),
            n_head=estimator_args.get("n_head", 4),
            scaling=estimator_args.get("scaling", "mean"),
            time_feat=estimator_args.get("time_feat", False),
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        self._estimator = estimator
        self._lightning_module = lightning_module
        self._transformation = transformation
        self._device = device
        self._is_fitted = True

        logger.info("Lag-Llama loaded from %s", self.ckpt_path)
        return self

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
        from gluonts.torch.model.predictor import PyTorchPredictor

        # Rebuild estimator with correct prediction_length
        self._estimator.prediction_length = prediction_length

        predictor = PyTorchPredictor(
            prediction_length=prediction_length,
            prediction_net=self._lightning_module,
            input_transform=self._transformation,
            batch_size=1,
            device=self._device,
            input_names=["past_target", "past_observed_values"],
        )

        # Build GluonTS dataset from context
        ctx = context.iloc[-self.context_length:]
        ds = PandasDataset.from_long_dataframe(
            pd.DataFrame({"target": ctx, "item_id": "load"}),
            target="target",
            item_id="item_id",
        )

        forecasts = list(predictor.predict(ds, num_samples=num_samples))
        if not forecasts:
            raise RuntimeError("Lag-Llama returned no forecasts")

        fc = forecasts[0]
        samples = fc.samples  # (num_samples, prediction_length)
        point_forecast = np.median(samples, axis=0)

        return point_forecast, samples
