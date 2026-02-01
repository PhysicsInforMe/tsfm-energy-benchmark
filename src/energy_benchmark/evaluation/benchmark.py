"""Benchmark runner: rolling-window evaluation across models and horizons."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..models.base import ForecastModel
from . import metrics as M

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for a single forecast window result."""

    model_name: str
    horizon: int
    context_length: int
    window_idx: int
    point_forecast: np.ndarray
    actual: np.ndarray
    samples: Optional[np.ndarray] = None
    elapsed_seconds: float = 0.0


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results across all models/horizons/windows."""

    records: List[Dict[str, Any]] = field(default_factory=list)
    forecasts: List[ForecastResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a summary DataFrame with one row per (model, horizon, context)."""
        return pd.DataFrame(self.records)


class BenchmarkRunner:
    """Execute rolling-window evaluation for a set of models.

    Args:
        models: List of fitted :class:`ForecastModel` instances.
        prediction_horizons: Forecast horizons to evaluate (in time-steps).
        context_lengths: Context window sizes to test.
        num_samples: Number of probabilistic samples per forecast.
        metric_names: Which metrics to compute. Supported:
            ``'mae'``, ``'rmse'``, ``'mase'``, ``'crps'``, ``'wql'``.
    """

    METRIC_FNS = {
        "mae": lambda yt, yp, **kw: M.mae(yt, yp),
        "rmse": lambda yt, yp, **kw: M.rmse(yt, yp),
        "mase": lambda yt, yp, **kw: M.mase(yt, yp, kw["y_train"], seasonality=24),
        "crps": lambda yt, yp, **kw: (
            M.crps(yt, kw["samples"]) if kw.get("samples") is not None else float("nan")
        ),
        "wql": lambda yt, yp, **kw: (
            M.weighted_quantile_loss(
                yt,
                np.quantile(kw["samples"], [0.1, 0.5, 0.9], axis=0).T,
            )
            if kw.get("samples") is not None
            else float("nan")
        ),
    }

    def __init__(
        self,
        models: List[ForecastModel],
        prediction_horizons: List[int] | None = None,
        context_lengths: List[int] | None = None,
        num_samples: int = 100,
        metric_names: List[str] | None = None,
    ) -> None:
        self.models = models
        self.prediction_horizons = prediction_horizons or [24, 168, 720]
        self.context_lengths = context_lengths or [512]
        self.num_samples = num_samples
        self.metric_names = metric_names or ["mae", "rmse", "mase"]

    def run(
        self,
        train: pd.Series,
        test: pd.Series,
        rolling_config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResults:
        """Run the full benchmark.

        Args:
            train: Training series (used for fitting and MASE scaling).
            test: Test series to evaluate on.
            rolling_config: Dict with keys ``step_size`` and ``num_windows``.

        Returns:
            :class:`BenchmarkResults` with per-window forecasts and
            aggregated metric records.
        """
        step_size = 24
        num_windows = 30
        if rolling_config:
            step_size = rolling_config.get("step_size", step_size)
            num_windows = rolling_config.get("num_windows", num_windows)

        results = BenchmarkResults()
        y_train = train.values

        for model in self.models:
            if not model._is_fitted:
                logger.info("Fitting %s...", model.name)
                model.fit(train)

            for ctx_len in self.context_lengths:
                for horizon in self.prediction_horizons:
                    window_metrics = self._evaluate_rolling(
                        model=model,
                        train=train,
                        test=test,
                        context_length=ctx_len,
                        horizon=horizon,
                        step_size=step_size,
                        num_windows=num_windows,
                        y_train=y_train,
                        results=results,
                    )

                    # Aggregate across windows
                    agg = {
                        "model": model.name,
                        "horizon": horizon,
                        "context_length": ctx_len,
                        "num_windows": len(window_metrics),
                    }
                    for metric_name in self.metric_names:
                        vals = [w[metric_name] for w in window_metrics if not np.isnan(w[metric_name])]
                        agg[metric_name] = float(np.mean(vals)) if vals else float("nan")

                    time_vals = [w["elapsed_seconds"] for w in window_metrics]
                    agg["mean_inference_seconds"] = float(np.mean(time_vals))

                    results.records.append(agg)
                    logger.info(
                        "%s | horizon=%d ctx=%d | %s",
                        model.name,
                        horizon,
                        ctx_len,
                        {k: f"{v:.4f}" for k, v in agg.items() if isinstance(v, float)},
                    )

        return results

    def _evaluate_rolling(
        self,
        model: ForecastModel,
        train: pd.Series,
        test: pd.Series,
        context_length: int,
        horizon: int,
        step_size: int,
        num_windows: int,
        y_train: np.ndarray,
        results: BenchmarkResults,
    ) -> List[Dict[str, float]]:
        """Evaluate a single model on rolling windows."""
        full_series = pd.concat([train, test])
        test_start_idx = len(train)
        window_metrics: List[Dict[str, float]] = []

        max_windows = min(
            num_windows,
            max(1, (len(test) - horizon) // step_size),
        )

        desc = f"{model.name} h={horizon} ctx={context_length}"
        for w in tqdm(range(max_windows), desc=desc, leave=False):
            # The point in test where this window's forecast begins
            forecast_start = test_start_idx + w * step_size
            forecast_end = forecast_start + horizon

            if forecast_end > len(full_series):
                break

            # Context is the window preceding the forecast start
            ctx_start = max(0, forecast_start - context_length)
            context = full_series.iloc[ctx_start:forecast_start]
            actual = full_series.iloc[forecast_start:forecast_end].values

            t0 = time.perf_counter()
            try:
                point, samples = model.predict(
                    context,
                    prediction_length=horizon,
                    num_samples=self.num_samples,
                )
            except Exception as e:
                logger.warning(
                    "Window %d failed for %s: %s", w, model.name, e
                )
                continue
            elapsed = time.perf_counter() - t0

            fr = ForecastResult(
                model_name=model.name,
                horizon=horizon,
                context_length=context_length,
                window_idx=w,
                point_forecast=point,
                actual=actual,
                samples=samples,
                elapsed_seconds=elapsed,
            )
            results.forecasts.append(fr)

            # Compute metrics for this window
            m = {}
            for name in self.metric_names:
                fn = self.METRIC_FNS.get(name)
                if fn is None:
                    m[name] = float("nan")
                    continue
                try:
                    m[name] = fn(
                        actual,
                        point,
                        y_train=y_train,
                        samples=samples,
                    )
                except Exception as e:
                    logger.debug("Metric %s failed: %s", name, e)
                    m[name] = float("nan")
            m["elapsed_seconds"] = elapsed
            window_metrics.append(m)

        return window_metrics
