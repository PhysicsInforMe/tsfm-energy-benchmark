"""Visualization utilities for forecast comparison and analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_comparison(
    results_df: pd.DataFrame,
    metric: str = "mae",
    save_path: Optional[str | Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Bar chart comparing models across horizons for a given metric.

    Args:
        results_df: DataFrame from :meth:`BenchmarkResults.to_dataframe`.
        metric: Column name of the metric to plot.
        save_path: If provided, save figure to this path.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    pivot = results_df.pivot_table(
        index="model", columns="horizon", values=metric, aggfunc="mean"
    )
    pivot.plot(kind="bar", ax=ax)

    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison — {metric.upper()} by Forecast Horizon")
    ax.set_xlabel("")
    ax.legend(title="Horizon (hours)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_forecasts(
    actual: pd.Series,
    predictions: Dict[str, np.ndarray],
    start_idx: int = 0,
    length: int = 168,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot actual vs predicted time series for multiple models.

    Args:
        actual: Ground truth series.
        predictions: Mapping ``{model_name: point_forecast_array}``.
        start_idx: Starting index into the actual series.
        length: Number of time-steps to display.
        save_path: If provided, save figure to this path.
        figsize: Matplotlib figure size.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    end_idx = min(start_idx + length, len(actual))
    segment = actual.iloc[start_idx:end_idx]

    ax.plot(segment.index, segment.values, "k-", linewidth=1.5, label="Actual")

    for name, pred in predictions.items():
        pred_segment = pred[start_idx:end_idx]
        ax.plot(
            segment.index[: len(pred_segment)],
            pred_segment,
            linewidth=1,
            label=name,
            alpha=0.8,
        )

    ax.set_ylabel("Load (MW)")
    ax.set_title("Forecast vs Actual")
    ax.legend(loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metric_heatmap(
    results_df: pd.DataFrame,
    metric: str = "mase",
    save_path: Optional[str | Path] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Heatmap of a metric across models and horizons.

    Args:
        results_df: Benchmark results DataFrame.
        metric: Metric column to display.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    pivot = results_df.pivot_table(
        index="model", columns="horizon", values=metric, aggfunc="mean"
    )
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
    )
    ax.set_title(f"{metric.upper()} — Model x Horizon")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_probabilistic_forecast(
    actual: np.ndarray,
    point_forecast: np.ndarray,
    samples: Optional[np.ndarray] = None,
    quantiles: tuple = (0.1, 0.9),
    save_path: Optional[str | Path] = None,
    figsize: tuple = (12, 5),
    title: str = "Probabilistic Forecast",
) -> plt.Figure:
    """Plot a single probabilistic forecast with uncertainty bands.

    Args:
        actual: Ground truth values.
        point_forecast: Median / point prediction.
        samples: Shape ``(num_samples, prediction_length)``.
        quantiles: Lower/upper quantile levels for the band.
        save_path: Optional save path.
        figsize: Figure size.
        title: Plot title.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(actual))

    ax.plot(x, actual, "k-", linewidth=1.5, label="Actual")
    ax.plot(x, point_forecast, "b-", linewidth=1, label="Forecast (median)")

    if samples is not None:
        lower = np.quantile(samples, quantiles[0], axis=0)
        upper = np.quantile(samples, quantiles[1], axis=0)
        ax.fill_between(
            x[: len(lower)],
            lower,
            upper,
            alpha=0.25,
            color="blue",
            label=f"{int(quantiles[0]*100)}–{int(quantiles[1]*100)}% CI",
        )

    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("Hour")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
