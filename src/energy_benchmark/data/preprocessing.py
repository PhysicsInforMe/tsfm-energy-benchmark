"""Preprocessing utilities for time series data."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def preprocess_series(
    series: pd.Series,
    fill_method: str = "linear",
    clip_std: float | None = 5.0,
) -> pd.Series:
    """Clean and preprocess a load series.

    Args:
        series: Raw hourly load series with DatetimeIndex.
        fill_method: Method for filling missing values
            (``"linear"``, ``"ffill"``, or ``"zero"``).
        clip_std: If set, clip values beyond this many standard deviations
            from the mean. Useful for removing obvious data errors.

    Returns:
        Cleaned series with no NaN values.
    """
    s = series.copy()

    # Fill missing values
    if fill_method == "linear":
        s = s.interpolate(method="time")
    elif fill_method == "ffill":
        s = s.ffill()
    elif fill_method == "zero":
        s = s.fillna(0.0)

    # Back-fill any remaining NaN at the start
    s = s.bfill()

    # Clip extreme outliers
    if clip_std is not None:
        mean, std = s.mean(), s.std()
        lower = mean - clip_std * std
        upper = mean + clip_std * std
        s = s.clip(lower=lower, upper=upper)

    return s


def create_splits(
    series: pd.Series,
    train_end: str,
    val_end: str,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Split a series into train / val / test by date boundaries.

    Args:
        series: Full series with DatetimeIndex.
        train_end: Last date (inclusive) for training.
        val_end: Last date (inclusive) for validation.

    Returns:
        (train, val, test) tuple.
    """
    train = series[series.index <= pd.Timestamp(train_end)]
    val = series[
        (series.index > pd.Timestamp(train_end))
        & (series.index <= pd.Timestamp(val_end))
    ]
    test = series[series.index > pd.Timestamp(val_end)]
    return train, val, test


def normalize(
    series: pd.Series,
    method: str = "standard",
    params: dict | None = None,
) -> Tuple[pd.Series, dict]:
    """Normalize a series and return the parameters for inverse transform.

    Args:
        series: Input series.
        method: ``"standard"`` (zero-mean, unit-var) or ``"minmax"`` (0-1).
        params: If provided, use these parameters instead of computing
            from data (useful for normalizing test sets).

    Returns:
        (normalized_series, params_dict) where params_dict can be passed
        back to denormalize the output.
    """
    if method == "standard":
        if params is None:
            params = {"mean": series.mean(), "std": series.std(), "method": method}
        normalized = (series - params["mean"]) / params["std"]
    elif method == "minmax":
        if params is None:
            params = {"min": series.min(), "max": series.max(), "method": method}
        normalized = (series - params["min"]) / (params["max"] - params["min"])
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def denormalize(series: pd.Series, params: dict) -> pd.Series:
    """Reverse a normalization using stored parameters."""
    method = params["method"]
    if method == "standard":
        return series * params["std"] + params["mean"]
    elif method == "minmax":
        return series * (params["max"] - params["min"]) + params["min"]
    raise ValueError(f"Unknown normalization method: {method}")
