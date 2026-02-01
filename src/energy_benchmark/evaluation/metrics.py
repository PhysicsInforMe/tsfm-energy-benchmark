"""Evaluation metrics for point and probabilistic forecasts."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 24,
) -> float:
    """Mean Absolute Scaled Error.

    MASE < 1 means the model outperforms a seasonal naive baseline.

    Args:
        y_true: Actual values, shape ``(n,)``.
        y_pred: Predicted values, shape ``(n,)``.
        y_train: Training series used to compute the naive scaling factor.
        seasonality: Seasonal period (24 = daily for hourly data).

    Returns:
        MASE score (lower is better).
    """
    n = len(y_train)
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scaling_factor = naive_errors.mean()
    if scaling_factor < 1e-9:
        return float("inf")
    return float(np.mean(np.abs(y_true - y_pred)) / scaling_factor)


def crps(y_true: np.ndarray, forecast_samples: np.ndarray) -> float:
    """Continuous Ranked Probability Score (empirical).

    Args:
        y_true: Actual values, shape ``(n,)``.
        forecast_samples: Sampled forecasts, shape ``(num_samples, n)``.

    Returns:
        Mean CRPS (lower is better).
    """
    try:
        from properscoring import crps_ensemble

        return float(crps_ensemble(y_true, forecast_samples.T).mean())
    except ImportError:
        # Fallback: manual computation
        num_samples = forecast_samples.shape[0]
        n = len(y_true)
        scores = np.zeros(n)
        for i in range(n):
            samples = np.sort(forecast_samples[:, i])
            obs = y_true[i]
            # CRPS = E|X - y| - 0.5 * E|X - X'|
            term1 = np.mean(np.abs(samples - obs))
            term2 = np.mean(
                np.abs(samples[:, None] - samples[None, :])
            ) / 2.0
            scores[i] = term1 - term2
        return float(scores.mean())


def weighted_quantile_loss(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantiles: list[float] | None = None,
) -> float:
    """Weighted Quantile Loss (WQL).

    Args:
        y_true: Actual values, shape ``(n,)``.
        y_pred_quantiles: Predicted quantiles, shape ``(n, num_quantiles)``.
        quantiles: Quantile levels corresponding to columns of
            ``y_pred_quantiles``. Defaults to ``[0.1, 0.5, 0.9]``.

    Returns:
        Mean WQL (lower is better).
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred_quantiles[:, i]
        loss = np.where(errors >= 0, q * errors, (q - 1.0) * errors)
        losses.append(loss.mean())

    return float(np.mean(losses))
