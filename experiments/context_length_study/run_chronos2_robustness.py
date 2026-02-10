"""
Run Chronos-2 through Distribution Shift Analysis only.

This script runs Chronos-2 through the same analysis pipeline as the other models
to generate data for Table 3 (Distribution Shift).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def load_ercot_data():
    """Load ERCOT data."""
    from energy_benchmark.data import ERCOTLoader
    from energy_benchmark.data.preprocessing import preprocess_series

    print("Loading ERCOT data...")
    loader = ERCOTLoader(years=[2020, 2021, 2022, 2023, 2024])
    series = loader.load()
    series = preprocess_series(series)
    print(f"Loaded {len(series):,} observations")
    return series


def compute_metrics(actual: np.ndarray, predicted: np.ndarray, train_data: np.ndarray) -> dict:
    """Compute forecast metrics."""
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    # MASE with weekly seasonality
    seasonality = 168
    naive_errors = np.abs(train_data[seasonality:] - train_data[:-seasonality])
    scaling_factor = naive_errors.mean()
    mase = mae / scaling_factor if scaling_factor > 1e-9 else float("inf")

    return {"mae": mae, "rmse": rmse, "mase": mase}


def analyze_distribution_shift(
    series: pd.Series,
    model,
    model_name: str,
    train_data: np.ndarray,
    context_length: int = 512,
    horizon: int = 24,
    num_windows: int = 10,
) -> pd.DataFrame:
    """
    Analyze model performance under different distribution shifts.
    Same periods as robustness_analysis.py
    """
    periods = {
        "normal_summer": ("2021-07-01", "2021-07-31"),
        "covid_lockdown": ("2020-04-01", "2020-04-30"),
        "winter_storm": ("2021-02-10", "2021-02-28"),
        "recent_2023": ("2023-09-01", "2023-09-30"),
        "holiday_period": ("2022-12-20", "2023-01-05"),
    }

    results = []

    for period_name, (start_date, end_date) in periods.items():
        period_start = pd.Timestamp(start_date)

        try:
            period_start_idx = series.index.get_loc(period_start)
        except KeyError:
            print(f"  Period {period_name} not found in data")
            continue

        if period_start_idx < context_length:
            print(f"  Not enough history for {period_name}")
            continue

        print(f"  Running {period_name}...")
        for w in range(num_windows):
            forecast_start_idx = period_start_idx + w * 24

            if forecast_start_idx + horizon > len(series):
                break

            ctx_start = forecast_start_idx - context_length
            context = series.iloc[ctx_start:forecast_start_idx]
            actual = series.iloc[forecast_start_idx:forecast_start_idx + horizon].values

            if len(actual) < horizon:
                break

            try:
                point_forecast, _ = model.predict(context, prediction_length=horizon, num_samples=50)
            except Exception as e:
                print(f"    Window {w} failed: {e}")
                continue

            metrics = compute_metrics(actual, point_forecast, train_data)

            results.append({
                "model": model_name,
                "period": period_name,
                "window": w,
                **metrics,
            })

    return pd.DataFrame(results)


def main():
    output_dir = Path(__file__).parent / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    series = load_ercot_data()
    train = series[:"2022-12-31"]
    train_data = train.values

    # Load Chronos-2
    print("\nLoading Chronos-2...")
    try:
        from energy_benchmark.models.chronos2 import Chronos2Model
        model = Chronos2Model(device="cpu")
        # fit() loads the pre-trained model (zero-shot, no actual training)
        model.fit(train)
        print("  Chronos-2 loaded successfully")
    except Exception as e:
        print(f"  ERROR loading Chronos-2: {e}")
        return

    # Run distribution shift analysis
    print("\nRunning distribution shift analysis for Chronos-2...")
    results_df = analyze_distribution_shift(
        series, model, "Chronos-2", train_data,
        context_length=512, horizon=24, num_windows=10
    )

    if results_df.empty:
        print("ERROR: No results generated!")
        return

    # Compute summary
    print("\n" + "=" * 70)
    print("CHRONOS-2 DISTRIBUTION SHIFT RESULTS (C=512, H=24)")
    print("=" * 70)

    period_means = results_df.groupby("period")["mase"].mean()

    # Get normal_summer as reference
    if "normal_summer" in period_means.index:
        ref_mase = period_means["normal_summer"]
    else:
        ref_mase = period_means.iloc[0]
        print("WARNING: normal_summer not found, using first period as reference")

    print(f"\nNormal Summer (ref.): {ref_mase:.3f}")

    for period in ["covid_lockdown", "recent_2023", "winter_storm", "holiday_period"]:
        if period in period_means.index:
            mase = period_means[period]
            pct_change = ((mase - ref_mase) / ref_mase) * 100
            sign = "+" if pct_change > 0 else ""
            print(f"{period.replace('_', ' ').title():20s}: {mase:.3f} ({sign}{pct_change:.0f}%)")
        else:
            print(f"{period.replace('_', ' ').title():20s}: NOT AVAILABLE")

    # Save results
    results_df.to_csv(output_dir / "chronos2_distribution_shift.csv", index=False)
    print(f"\nSaved: {output_dir / 'chronos2_distribution_shift.csv'}")

    # Also append to the main robustness file if it exists
    main_file = output_dir / "robustness_distribution_shift.csv"
    if main_file.exists():
        existing = pd.read_csv(main_file)
        # Remove any existing Chronos-2 data
        existing = existing[existing["model"] != "Chronos-2"]
        combined = pd.concat([existing, results_df], ignore_index=True)
        combined.to_csv(main_file, index=False)
        print(f"Updated: {main_file}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    results = main()
