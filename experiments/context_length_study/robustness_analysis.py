"""
Robustness Analysis for Energy Load Forecasting Models

This module evaluates how well models perform under:
1. Distribution Shift (training vs test period differences)
2. Extreme Events (anomalous load patterns)
3. Missing Data / Data Quality Issues
4. Different Temporal Patterns (weekday vs weekend, seasonal)

These analyses help understand model reliability in real-world deployments.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Set publication-quality plot style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_ercot_data():
    """Load ERCOT data."""
    from energy_benchmark.data import ERCOTLoader
    from energy_benchmark.data.preprocessing import preprocess_series

    loader = ERCOTLoader(years=[2020, 2021, 2022, 2023, 2024])
    series = loader.load()
    series = preprocess_series(series)
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


# =============================================================================
# Distribution Shift Analysis
# =============================================================================

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

    Tests on:
    - Normal period (similar to training)
    - COVID period (anomalous demand patterns)
    - Extreme weather (high demand events)
    - Recent period (potential data drift)
    """
    periods = {
        "normal_summer": ("2021-07-01", "2021-07-31"),  # Normal summer (training period)
        "covid_lockdown": ("2020-04-01", "2020-04-30"),  # COVID lockdown
        "winter_storm": ("2021-02-10", "2021-02-28"),   # Texas winter storm (extreme)
        "recent_2023": ("2023-09-01", "2023-09-30"),    # Recent (potential drift)
        "holiday_period": ("2022-12-20", "2023-01-05"),  # Holiday period
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
                print(f"  Prediction failed for {period_name} window {w}: {e}")
                continue

            metrics = compute_metrics(actual, point_forecast, train_data)

            results.append({
                "model": model_name,
                "period": period_name,
                "window": w,
                **metrics,
            })

    return pd.DataFrame(results)


# =============================================================================
# Temporal Pattern Analysis
# =============================================================================

def analyze_temporal_patterns(
    series: pd.Series,
    model,
    model_name: str,
    train_data: np.ndarray,
    context_length: int = 512,
    horizon: int = 24,
    test_start: str = "2023-07-01",
    num_windows: int = 30,
) -> pd.DataFrame:
    """
    Analyze performance across different temporal patterns.

    Tests on:
    - Weekdays vs weekends
    - Morning vs evening peaks
    - Different seasons
    """
    test_start_ts = pd.Timestamp(test_start)
    test_start_idx = series.index.get_loc(test_start_ts)

    results = []

    for w in range(num_windows):
        forecast_start_idx = test_start_idx + w * 24

        if forecast_start_idx + horizon > len(series):
            break

        forecast_time = series.index[forecast_start_idx]

        # Classify the forecast period
        day_of_week = forecast_time.dayofweek
        is_weekend = day_of_week >= 5
        month = forecast_time.month
        hour = forecast_time.hour

        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:
            season = "fall"

        ctx_start = forecast_start_idx - context_length
        context = series.iloc[ctx_start:forecast_start_idx]
        actual = series.iloc[forecast_start_idx:forecast_start_idx + horizon].values

        if len(actual) < horizon:
            break

        try:
            point_forecast, _ = model.predict(context, prediction_length=horizon, num_samples=50)
        except Exception as e:
            continue

        metrics = compute_metrics(actual, point_forecast, train_data)

        results.append({
            "model": model_name,
            "window": w,
            "day_type": "weekend" if is_weekend else "weekday",
            "season": season,
            "forecast_hour": hour,
            **metrics,
        })

    return pd.DataFrame(results)


# =============================================================================
# Missing Data Robustness
# =============================================================================

def analyze_missing_data_robustness(
    series: pd.Series,
    model,
    model_name: str,
    train_data: np.ndarray,
    context_length: int = 512,
    horizon: int = 24,
    missing_rates: list[float] = [0.0, 0.05, 0.10, 0.20],
) -> pd.DataFrame:
    """
    Analyze model robustness to missing data in context.

    Simulates missing data by replacing values with interpolation.
    """
    test_start = pd.Timestamp("2023-08-01")
    test_start_idx = series.index.get_loc(test_start)

    results = []
    num_windows = 10

    for missing_rate in missing_rates:
        for w in range(num_windows):
            forecast_start_idx = test_start_idx + w * 24

            if forecast_start_idx + horizon > len(series):
                break

            ctx_start = forecast_start_idx - context_length
            context = series.iloc[ctx_start:forecast_start_idx].copy()
            actual = series.iloc[forecast_start_idx:forecast_start_idx + horizon].values

            if len(actual) < horizon:
                break

            # Simulate missing data
            if missing_rate > 0:
                n_missing = int(len(context) * missing_rate)
                missing_indices = np.random.choice(len(context), n_missing, replace=False)

                # Replace with NaN then interpolate
                context_modified = context.copy()
                context_modified.iloc[missing_indices] = np.nan
                context_modified = context_modified.interpolate(method='linear')
                context_modified = context_modified.ffill().bfill()
            else:
                context_modified = context

            try:
                point_forecast, _ = model.predict(context_modified, prediction_length=horizon, num_samples=50)
            except Exception as e:
                continue

            metrics = compute_metrics(actual, point_forecast, train_data)

            results.append({
                "model": model_name,
                "missing_rate": missing_rate,
                "window": w,
                **metrics,
            })

    return pd.DataFrame(results)


# =============================================================================
# Extreme Event Detection
# =============================================================================

def analyze_extreme_events(
    series: pd.Series,
    model,
    model_name: str,
    train_data: np.ndarray,
    context_length: int = 512,
    horizon: int = 24,
) -> pd.DataFrame:
    """
    Analyze model performance during extreme load events.

    Identifies periods with unusually high or low demand and tests performance.
    """
    # Find extreme events (top/bottom 5% of daily peaks)
    daily_peaks = series.resample('D').max()
    high_threshold = daily_peaks.quantile(0.95)
    low_threshold = daily_peaks.quantile(0.05)

    extreme_high_days = daily_peaks[daily_peaks > high_threshold].index
    extreme_low_days = daily_peaks[daily_peaks < low_threshold].index
    normal_days = daily_peaks[
        (daily_peaks <= high_threshold) & (daily_peaks >= low_threshold)
    ].index

    results = []

    for event_type, days in [("extreme_high", extreme_high_days),
                              ("extreme_low", extreme_low_days),
                              ("normal", normal_days[:30])]:  # Limit normal days

        for day in days[:10]:  # Limit to 10 windows per type
            forecast_start = pd.Timestamp(day) + pd.Timedelta(hours=12)  # Noon forecast

            try:
                forecast_start_idx = series.index.get_loc(forecast_start)
            except KeyError:
                continue

            if forecast_start_idx < context_length:
                continue
            if forecast_start_idx + horizon > len(series):
                continue

            ctx_start = forecast_start_idx - context_length
            context = series.iloc[ctx_start:forecast_start_idx]
            actual = series.iloc[forecast_start_idx:forecast_start_idx + horizon].values

            try:
                point_forecast, samples = model.predict(
                    context, prediction_length=horizon, num_samples=100
                )
            except Exception:
                continue

            metrics = compute_metrics(actual, point_forecast, train_data)

            # Additional metrics for extreme events
            if samples is not None:
                # Check if actual values fall within prediction intervals
                lower_90 = np.quantile(samples, 0.05, axis=0)
                upper_90 = np.quantile(samples, 0.95, axis=0)
                coverage = np.mean((actual >= lower_90) & (actual <= upper_90))
            else:
                coverage = None

            results.append({
                "model": model_name,
                "event_type": event_type,
                "date": str(day.date()),
                "coverage_90": coverage,
                **metrics,
            })

    return pd.DataFrame(results)


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_distribution_shift_results(df: pd.DataFrame, output_path: Path):
    """Plot distribution shift analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MASE by period
    ax = axes[0]
    period_means = df.groupby(["model", "period"])["mase"].mean().unstack()

    period_means.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("MASE")
    ax.set_xlabel("Model")
    ax.set_title("Performance Across Different Distribution Shifts")
    ax.legend(title="Period", loc="upper right")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Relative degradation
    ax = axes[1]
    if "normal_summer" in period_means.columns:
        baseline = period_means["normal_summer"]
        relative = period_means.div(baseline, axis=0) * 100 - 100

        relative.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("% Degradation from Normal")
        ax.set_xlabel("Model")
        ax.set_title("Performance Degradation Under Distribution Shift")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(title="Period", loc="upper right")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path / "fig10_distribution_shift.png")
    fig.savefig(output_path / "fig10_distribution_shift.pdf")
    plt.close()
    print(f"Saved: fig10_distribution_shift.png/pdf")


def plot_temporal_pattern_results(df: pd.DataFrame, output_path: Path):
    """Plot temporal pattern analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Weekday vs Weekend
    ax = axes[0]
    day_means = df.groupby(["model", "day_type"])["mase"].mean().unstack()
    day_means.plot(kind="bar", ax=ax, width=0.6)
    ax.set_ylabel("MASE")
    ax.set_xlabel("Model")
    ax.set_title("Performance: Weekday vs Weekend")
    ax.legend(title="Day Type")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # By Season
    ax = axes[1]
    season_means = df.groupby(["model", "season"])["mase"].mean().unstack()
    if not season_means.empty:
        season_means.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("MASE")
        ax.set_xlabel("Model")
        ax.set_title("Performance Across Seasons")
        ax.legend(title="Season")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path / "fig11_temporal_patterns.png")
    fig.savefig(output_path / "fig11_temporal_patterns.pdf")
    plt.close()
    print(f"Saved: fig11_temporal_patterns.png/pdf")


def plot_missing_data_results(df: pd.DataFrame, output_path: Path):
    """Plot missing data robustness results."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        means = model_data.groupby("missing_rate")["mase"].mean()
        stds = model_data.groupby("missing_rate")["mase"].std()

        ax.errorbar(means.index * 100, means.values, yerr=stds.values,
                   marker='o', capsize=5, label=model, linewidth=2)

    ax.set_xlabel("Missing Data Rate (%)")
    ax.set_ylabel("MASE")
    ax.set_title("Model Robustness to Missing Data")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path / "fig12_missing_data.png")
    fig.savefig(output_path / "fig12_missing_data.pdf")
    plt.close()
    print(f"Saved: fig12_missing_data.png/pdf")


def plot_extreme_event_results(df: pd.DataFrame, output_path: Path):
    """Plot extreme event analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MASE by event type
    ax = axes[0]
    event_means = df.groupby(["model", "event_type"])["mase"].mean().unstack()
    event_means.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("MASE")
    ax.set_xlabel("Model")
    ax.set_title("Performance During Extreme Events")
    ax.legend(title="Event Type")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Coverage during extreme events
    ax = axes[1]
    if "coverage_90" in df.columns:
        coverage_means = df.groupby(["model", "event_type"])["coverage_90"].mean().unstack()
        if not coverage_means.empty:
            coverage_means.plot(kind="bar", ax=ax, width=0.8)
            ax.axhline(0.9, color="red", linestyle="--", label="Target 90%")
            ax.set_ylabel("90% PI Coverage")
            ax.set_xlabel("Model")
            ax.set_title("Uncertainty Calibration During Extreme Events")
            ax.legend(title="Event Type")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path / "fig13_extreme_events.png")
    fig.savefig(output_path / "fig13_extreme_events.pdf")
    plt.close()
    print(f"Saved: fig13_extreme_events.png/pdf")


def create_robustness_summary(
    dist_shift_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    extreme_df: pd.DataFrame,
    output_path: Path,
):
    """Create summary table of robustness analysis."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print("=" * 70)

    # Distribution shift
    print("\n1. Distribution Shift Analysis:")
    print("-" * 50)
    dist_summary = dist_shift_df.groupby(["model", "period"])["mase"].mean().unstack()
    print(dist_summary.round(3).to_string())

    # Temporal patterns
    print("\n2. Temporal Pattern Analysis:")
    print("-" * 50)
    temporal_summary = temporal_df.groupby(["model", "day_type"])["mase"].mean().unstack()
    print(temporal_summary.round(3).to_string())

    # Missing data
    print("\n3. Missing Data Robustness:")
    print("-" * 50)
    missing_summary = missing_df.groupby(["model", "missing_rate"])["mase"].mean().unstack()
    print(missing_summary.round(3).to_string())

    # Extreme events
    print("\n4. Extreme Event Performance:")
    print("-" * 50)
    extreme_summary = extreme_df.groupby(["model", "event_type"])["mase"].mean().unstack()
    print(extreme_summary.round(3).to_string())

    # Save combined results
    all_results = {
        "distribution_shift": dist_shift_df,
        "temporal_patterns": temporal_df,
        "missing_data": missing_df,
        "extreme_events": extreme_df,
    }

    for name, df in all_results.items():
        df.to_csv(output_path / f"robustness_{name}.csv", index=False)

    print(f"\nSaved: robustness_*.csv files")


def main():
    """Run robustness analysis."""
    output_dir = Path(__file__).parent / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    series = load_ercot_data()
    train = series[:"2022-12-31"]
    train_data = train.values

    print("Loading models...")
    models = {}

    # Load foundation models
    model_configs = [
        ("Chronos-Bolt", "chronos_bolt", {"model_size": "small", "device": "cpu"}),
        ("Chronos-2", "chronos2", {"device": "cpu"}),
        ("Moirai-2", "moirai", {"model_type": "moirai2", "size": "small", "device": "cpu"}),
        ("TTM", "tinytimemixer", {"model_version": "r2", "device": "cpu"}),
        ("Prophet", "prophet", {"weekly_seasonality": True, "daily_seasonality": True}),
    ]

    for name, model_type, config in model_configs:
        try:
            if model_type == "chronos_bolt":
                from energy_benchmark.models.chronos_bolt import ChronosBoltModel
                model = ChronosBoltModel(**config)
            elif model_type == "chronos2":
                from energy_benchmark.models.chronos2 import Chronos2Model
                model = Chronos2Model(**config)
            elif model_type == "moirai":
                from energy_benchmark.models.moirai import MoiraiModel
                model = MoiraiModel(**config)
            elif model_type == "tinytimemixer":
                from energy_benchmark.models.tinytimemixer import TinyTimeMixerModel
                model = TinyTimeMixerModel(**config)
            elif model_type == "prophet":
                from energy_benchmark.models.prophet_model import ProphetModel
                model = ProphetModel(**config)

            if hasattr(model, 'fit'):
                model.fit(train)
            models[name] = model
            print(f"  Loaded {name}")
        except Exception as e:
            print(f"  Could not load {name}: {e}")

    if not models:
        print("No models loaded!")
        return

    # Run analyses
    all_dist_shift = []
    all_temporal = []
    all_missing = []
    all_extreme = []

    for model_name, model in models.items():
        print(f"\nAnalyzing {model_name}...")

        print("  Distribution shift analysis...")
        df = analyze_distribution_shift(series, model, model_name, train_data)
        all_dist_shift.append(df)

        print("  Temporal pattern analysis...")
        df = analyze_temporal_patterns(series, model, model_name, train_data)
        all_temporal.append(df)

        print("  Missing data robustness...")
        df = analyze_missing_data_robustness(series, model, model_name, train_data)
        all_missing.append(df)

        print("  Extreme event analysis...")
        df = analyze_extreme_events(series, model, model_name, train_data)
        all_extreme.append(df)

    # Combine results
    dist_shift_df = pd.concat(all_dist_shift, ignore_index=True)
    temporal_df = pd.concat(all_temporal, ignore_index=True)
    missing_df = pd.concat(all_missing, ignore_index=True)
    extreme_df = pd.concat(all_extreme, ignore_index=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    if not dist_shift_df.empty:
        plot_distribution_shift_results(dist_shift_df, output_dir)
    if not temporal_df.empty:
        plot_temporal_pattern_results(temporal_df, output_dir)
    if not missing_df.empty:
        plot_missing_data_results(missing_df, output_dir)
    if not extreme_df.empty:
        plot_extreme_event_results(extreme_df, output_dir)

    # Create summary
    create_robustness_summary(
        dist_shift_df, temporal_df, missing_df, extreme_df, output_dir
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
