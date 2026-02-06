"""
Context Length Sensitivity Experiment

Run the full experimental protocol to measure how context length
affects forecast accuracy across foundation models and baselines.

Usage:
    python run_experiment.py                    # Full experiment
    python run_experiment.py --test             # Quick test (1 condition)
    python run_experiment.py --context 512      # Single context length
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Experimental Configuration
# =============================================================================

CONTEXT_LENGTHS = [24, 48, 96, 168, 336, 512, 1024, 2048]
HORIZONS = [24, 168]
NUM_WINDOWS = 10
WINDOW_STEP = 24  # hours between rolling windows

# Test periods: (name, start_date, end_date)
TEST_PERIODS = [
    ("summer_2023", "2023-07-01", "2023-08-31"),
    ("winter_2022", "2022-12-01", "2023-02-28"),
    ("covid_2020", "2020-03-15", "2020-04-30"),
]

# Models to evaluate
MODEL_CONFIGS = [
    ("SeasonalNaive", "seasonal_naive", {"seasonality": 168}),
    ("SARIMA", "arima", {"order": (2, 1, 2), "seasonal_order": (1, 1, 1, 24)}),
    ("Chronos-Bolt", "chronos_bolt", {"model_size": "small", "device": "cpu"}),
    ("Chronos-2", "chronos2", {"device": "cpu"}),
    ("Lag-Llama", "lag_llama", {"ckpt_path": "lag-llama/lag-llama.ckpt", "device": "cpu"}),
]


@dataclass
class ForecastResult:
    """Single forecast result."""
    model: str
    context_length: int
    horizon: int
    period: str
    window_idx: int
    mae: float
    rmse: float
    mase: float
    inference_seconds: float
    timestamp: str


# =============================================================================
# Data Loading
# =============================================================================

def load_data() -> pd.Series:
    """Load and preprocess ERCOT data."""
    from energy_benchmark.data import ERCOTLoader
    from energy_benchmark.data.preprocessing import preprocess_series

    logger.info("Loading ERCOT data...")
    loader = ERCOTLoader(years=[2020, 2021, 2022, 2023, 2024])
    series = loader.load()
    series = preprocess_series(series)
    logger.info(f"Loaded {len(series):,} observations")
    return series


def get_train_data(series: pd.Series) -> pd.Series:
    """Get training data for MASE scaling (2020-2022)."""
    return series[:"2022-12-31"]


# =============================================================================
# Model Factory
# =============================================================================

def create_model(model_type: str, config: dict):
    """Create and return a model instance."""
    if model_type == "seasonal_naive":
        from energy_benchmark.models import SeasonalNaiveModel
        return SeasonalNaiveModel(**config)
    elif model_type == "arima":
        from energy_benchmark.models import ARIMAModel
        return ARIMAModel(**config)
    elif model_type == "chronos_bolt":
        from energy_benchmark.models.chronos_bolt import ChronosBoltModel
        return ChronosBoltModel(**config)
    elif model_type == "chronos2":
        from energy_benchmark.models.chronos2 import Chronos2Model
        return Chronos2Model(**config)
    elif model_type == "lag_llama":
        import sys
        sys.path.insert(0, "lag-llama")
        from energy_benchmark.models.lag_llama import LagLlamaModel
        return LagLlamaModel(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_data: np.ndarray,
    seasonality: int = 168,
) -> dict:
    """Compute forecast metrics."""
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    # MASE with weekly seasonality
    naive_errors = np.abs(train_data[seasonality:] - train_data[:-seasonality])
    scaling_factor = naive_errors.mean()
    mase = mae / scaling_factor if scaling_factor > 1e-9 else float("inf")

    return {"mae": mae, "rmse": rmse, "mase": mase}


# =============================================================================
# Experiment Runner
# =============================================================================

def run_single_condition(
    model,
    model_name: str,
    series: pd.Series,
    train_data: np.ndarray,
    context_length: int,
    horizon: int,
    period_name: str,
    period_start: str,
    period_end: str,
    num_windows: int = NUM_WINDOWS,
) -> list[ForecastResult]:
    """Run evaluation for a single experimental condition."""

    # Get period data
    period_data = series[period_start:period_end]
    if len(period_data) < horizon + context_length:
        logger.warning(
            f"Period {period_name} too short for ctx={context_length}, h={horizon}"
        )
        return []

    # Calculate valid window range
    # We need context_length hours before and horizon hours after
    period_start_ts = pd.Timestamp(period_start)

    # Find where we have enough context before the period
    earliest_forecast = series.index.get_loc(period_start_ts)
    if earliest_forecast < context_length:
        logger.warning(f"Not enough history for context={context_length}")
        return []

    results = []

    for w in range(num_windows):
        # Forecast start point within the period
        forecast_offset = w * WINDOW_STEP
        if forecast_offset + horizon > len(period_data):
            break

        forecast_start_idx = earliest_forecast + forecast_offset

        # Get context window (before forecast start)
        ctx_start_idx = forecast_start_idx - context_length
        context = series.iloc[ctx_start_idx:forecast_start_idx]

        # Get actual values
        actual = series.iloc[forecast_start_idx:forecast_start_idx + horizon].values

        if len(actual) < horizon:
            break

        # Run inference with timing
        t0 = time.perf_counter()
        try:
            point_forecast, _ = model.predict(
                context, prediction_length=horizon, num_samples=50
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            continue
        inference_time = time.perf_counter() - t0

        # Compute metrics
        metrics = compute_metrics(actual, point_forecast, train_data)

        results.append(ForecastResult(
            model=model_name,
            context_length=context_length,
            horizon=horizon,
            period=period_name,
            window_idx=w,
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            mase=metrics["mase"],
            inference_seconds=inference_time,
            timestamp=datetime.now().isoformat(),
        ))

    return results


def run_experiment(
    context_lengths: list[int] = CONTEXT_LENGTHS,
    horizons: list[int] = HORIZONS,
    test_periods: list[tuple] = TEST_PERIODS,
    model_configs: list[tuple] = MODEL_CONFIGS,
    output_dir: Path = Path("results/raw"),
) -> pd.DataFrame:
    """Run the full experiment."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    series = load_data()
    train_data = get_train_data(series).values

    all_results = []

    # Calculate total iterations for progress bar
    total = len(model_configs) * len(context_lengths) * len(horizons) * len(test_periods)

    with tqdm(total=total, desc="Experiment Progress") as pbar:
        for model_name, model_type, model_config in model_configs:
            logger.info(f"Loading model: {model_name}")
            model = create_model(model_type, model_config)
            model.fit(series[:"2022-12-31"])  # Fit on training data

            for context_length in context_lengths:
                for horizon in horizons:
                    for period_name, period_start, period_end in test_periods:
                        pbar.set_description(
                            f"{model_name} ctx={context_length} h={horizon} {period_name}"
                        )

                        results = run_single_condition(
                            model=model,
                            model_name=model_name,
                            series=series,
                            train_data=train_data,
                            context_length=context_length,
                            horizon=horizon,
                            period_name=period_name,
                            period_start=period_start,
                            period_end=period_end,
                        )

                        all_results.extend(results)
                        pbar.update(1)

            # Save intermediate results after each model
            df = pd.DataFrame([asdict(r) for r in all_results])
            df.to_csv(output_dir / "results_partial.csv", index=False)

    # Save final results
    df = pd.DataFrame([asdict(r) for r in all_results])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(output_dir / f"results_{timestamp}.csv", index=False)
    df.to_csv(output_dir / "results_latest.csv", index=False)

    logger.info(f"Saved {len(df)} results to {output_dir}")
    return df


# =============================================================================
# Quick Test
# =============================================================================

def run_quick_test():
    """Run a minimal test to verify everything works."""
    logger.info("Running quick test...")

    series = load_data()
    train_data = get_train_data(series).values

    # Test with one model, one context, one horizon, one period
    from energy_benchmark.models import SeasonalNaiveModel
    model = SeasonalNaiveModel(seasonality=168)
    model.fit(series[:"2022-12-31"])

    results = run_single_condition(
        model=model,
        model_name="SeasonalNaive",
        series=series,
        train_data=train_data,
        context_length=168,
        horizon=24,
        period_name="summer_2023",
        period_start="2023-07-01",
        period_end="2023-08-31",
        num_windows=3,
    )

    print("\n=== QUICK TEST RESULTS ===")
    for r in results:
        print(f"  Window {r.window_idx}: MAE={r.mae:.0f}, MASE={r.mase:.3f}, Time={r.inference_seconds:.3f}s")

    if results:
        avg_mase = np.mean([r.mase for r in results])
        print(f"\n  Average MASE: {avg_mase:.3f}")
        print("  (Should be ~1.0 for SeasonalNaive with same seasonality)")

        if 0.5 < avg_mase < 2.0:
            print("\n  [PASS] Results look reasonable!")
            return True
        else:
            print("\n  [WARN] MASE outside expected range")
            return False
    else:
        print("\n  [FAIL] No results generated")
        return False


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context Length Sensitivity Experiment")
    parser.add_argument("--test", action="store_true", help="Run quick test only")
    parser.add_argument("--context", type=int, help="Run single context length only")
    parser.add_argument("--model", type=str, help="Run single model only")
    args = parser.parse_args()

    if args.test:
        success = run_quick_test()
        exit(0 if success else 1)

    # Filter configurations if specified
    context_lengths = [args.context] if args.context else CONTEXT_LENGTHS
    model_configs = MODEL_CONFIGS
    if args.model:
        model_configs = [m for m in MODEL_CONFIGS if m[0] == args.model]
        if not model_configs:
            logger.error(f"Unknown model: {args.model}")
            exit(1)

    # Run experiment
    output_dir = Path(__file__).parent / "results" / "raw"
    df = run_experiment(
        context_lengths=context_lengths,
        model_configs=model_configs,
        output_dir=output_dir,
    )

    print(f"\n=== EXPERIMENT COMPLETE ===")
    print(f"Total results: {len(df)}")
    print(f"Results saved to: {output_dir}")
