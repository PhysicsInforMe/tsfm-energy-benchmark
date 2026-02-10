"""
Run Prophet model experiments only and merge with existing results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration
CONTEXT_LENGTHS = [24, 48, 96, 168, 336, 512, 1024, 2048]
HORIZONS = [24, 168]
NUM_WINDOWS = 10
WINDOW_STEP = 24

TEST_PERIODS = [
    ("summer_2023", "2023-07-01", "2023-08-31"),
    ("winter_2022", "2022-12-01", "2023-02-28"),
    ("covid_2020", "2020-03-15", "2020-04-30"),
]


@dataclass
class ForecastResult:
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


def load_data():
    from energy_benchmark.data import ERCOTLoader
    from energy_benchmark.data.preprocessing import preprocess_series

    logger.info("Loading ERCOT data...")
    loader = ERCOTLoader(years=[2020, 2021, 2022, 2023, 2024])
    series = loader.load()
    series = preprocess_series(series)
    logger.info(f"Loaded {len(series):,} observations")
    return series


def compute_metrics(actual, predicted, train_data, seasonality=168):
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    naive_errors = np.abs(train_data[seasonality:] - train_data[:-seasonality])
    scaling_factor = naive_errors.mean()
    mase = mae / scaling_factor if scaling_factor > 1e-9 else float("inf")
    return {"mae": mae, "rmse": rmse, "mase": mase}


def run_prophet_experiment():
    output_dir = Path("results/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    series = load_data()
    train_data = series[:"2022-12-31"].values

    # Load Prophet model
    logger.info("Loading Prophet model...")
    from energy_benchmark.models.prophet_model import ProphetModel
    model = ProphetModel(weekly_seasonality=True, daily_seasonality=True)

    results = []
    total_conditions = len(CONTEXT_LENGTHS) * len(HORIZONS) * len(TEST_PERIODS)

    pbar = tqdm(total=total_conditions, desc="Prophet experiments")

    for context_length in CONTEXT_LENGTHS:
        for horizon in HORIZONS:
            for period_name, period_start, period_end in TEST_PERIODS:
                pbar.set_description(f"Prophet ctx={context_length} h={horizon} {period_name}")

                # Get period data
                try:
                    period_data = series[period_start:period_end]
                    if len(period_data) < horizon + context_length:
                        logger.warning(f"Period {period_name} too short for ctx={context_length}, h={horizon}")
                        pbar.update(1)
                        continue

                    period_start_ts = pd.Timestamp(period_start)
                    earliest_forecast = series.index.get_loc(period_start_ts)

                    if earliest_forecast < context_length:
                        logger.warning(f"Not enough history for context={context_length}")
                        pbar.update(1)
                        continue

                    for w in range(NUM_WINDOWS):
                        forecast_offset = w * WINDOW_STEP
                        if forecast_offset + horizon > len(period_data):
                            break

                        forecast_start_idx = earliest_forecast + forecast_offset
                        ctx_start_idx = forecast_start_idx - context_length
                        context = series.iloc[ctx_start_idx:forecast_start_idx]
                        actual = series.iloc[forecast_start_idx:forecast_start_idx + horizon].values

                        if len(actual) < horizon:
                            break

                        t0 = time.perf_counter()
                        try:
                            point_forecast, _ = model.predict(
                                context, prediction_length=horizon, num_samples=50
                            )
                        except Exception as e:
                            logger.error(f"Prediction failed: {e}")
                            continue
                        inference_time = time.perf_counter() - t0

                        metrics = compute_metrics(actual, point_forecast, train_data)

                        results.append(ForecastResult(
                            model="Prophet",
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

                except Exception as e:
                    logger.error(f"Error in condition: {e}")

                pbar.update(1)

    pbar.close()

    # Save Prophet results
    prophet_df = pd.DataFrame([asdict(r) for r in results])
    prophet_df.to_csv(output_dir / "results_prophet.csv", index=False)
    logger.info(f"Saved {len(prophet_df)} Prophet results")

    # Merge with existing results
    existing_file = output_dir / "results_all_models.csv"
    if existing_file.exists():
        existing_df = pd.read_csv(existing_file)
        # Remove any old Prophet results
        existing_df = existing_df[existing_df["model"] != "Prophet"]
        # Combine
        combined_df = pd.concat([existing_df, prophet_df], ignore_index=True)
        combined_df.to_csv(output_dir / "results_all_models.csv", index=False)
        logger.info(f"Merged results: {len(combined_df)} total rows")
    else:
        prophet_df.to_csv(output_dir / "results_all_models.csv", index=False)

    return prophet_df


if __name__ == "__main__":
    run_prophet_experiment()
