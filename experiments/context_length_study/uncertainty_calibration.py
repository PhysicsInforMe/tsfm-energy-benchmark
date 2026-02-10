"""
Uncertainty Calibration Analysis

Analyze whether probabilistic forecasts are well-calibrated:
- Do 90% prediction intervals contain 90% of observations?
- Are prediction intervals too wide or too narrow?

This analysis uses the probabilistic samples from foundation models
to assess uncertainty quantification quality.
"""

from __future__ import annotations

import warnings
from pathlib import Path

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


def compute_calibration_metrics(
    actual: np.ndarray,
    samples: np.ndarray,
    quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
) -> dict:
    """
    Compute calibration metrics for probabilistic forecasts.

    Args:
        actual: Ground truth values (horizon,)
        samples: Probabilistic samples (num_samples, horizon)
        quantile_levels: Quantile levels to evaluate

    Returns:
        Dictionary with calibration metrics
    """
    metrics = {}

    # Compute empirical coverage for different confidence levels
    for alpha in [0.5, 0.8, 0.9, 0.95]:
        lower_q = (1 - alpha) / 2
        upper_q = 1 - lower_q

        lower = np.quantile(samples, lower_q, axis=0)
        upper = np.quantile(samples, upper_q, axis=0)

        coverage = np.mean((actual >= lower) & (actual <= upper))
        metrics[f"coverage_{int(alpha*100)}"] = coverage

        # Interval width (normalized by actual mean)
        width = np.mean(upper - lower) / np.mean(np.abs(actual))
        metrics[f"width_{int(alpha*100)}"] = width

    # Continuous Ranked Probability Score (CRPS)
    # Using empirical approximation
    crps_values = []
    for t in range(len(actual)):
        sorted_samples = np.sort(samples[:, t])
        n = len(sorted_samples)

        # CRPS = E|X - y| - 0.5 * E|X - X'|
        term1 = np.mean(np.abs(sorted_samples - actual[t]))
        term2 = np.mean(np.abs(sorted_samples[:, None] - sorted_samples[None, :]))
        crps_values.append(term1 - 0.5 * term2)

    metrics["crps"] = np.mean(crps_values)

    # Prediction interval score (Winkler score) for 90% interval
    alpha = 0.1
    lower = np.quantile(samples, alpha/2, axis=0)
    upper = np.quantile(samples, 1 - alpha/2, axis=0)

    width = upper - lower
    penalty_lower = (2/alpha) * (lower - actual) * (actual < lower)
    penalty_upper = (2/alpha) * (actual - upper) * (actual > upper)

    winkler = np.mean(width + penalty_lower + penalty_upper)
    metrics["winkler_90"] = winkler

    return metrics


def run_calibration_experiment(
    series: pd.Series,
    model_name: str,
    model,
    context_length: int = 512,
    horizon: int = 24,
    num_windows: int = 20,
    num_samples: int = 100,
) -> pd.DataFrame:
    """Run calibration experiment for a single model."""

    results = []

    # Use test period (2023)
    test_start_idx = series.index.get_loc(pd.Timestamp("2023-07-01"))

    for w in range(num_windows):
        forecast_start = test_start_idx + w * 24

        if forecast_start + horizon > len(series):
            break

        # Get context and actual
        context = series.iloc[forecast_start - context_length:forecast_start]
        actual = series.iloc[forecast_start:forecast_start + horizon].values

        if len(actual) < horizon:
            break

        # Get forecast
        try:
            point, samples = model.predict(context, prediction_length=horizon, num_samples=num_samples)
        except Exception as e:
            print(f"Window {w} failed: {e}")
            continue

        if samples is None:
            print(f"Window {w}: No samples returned")
            continue

        # Compute metrics
        metrics = compute_calibration_metrics(actual, samples)
        metrics["model"] = model_name
        metrics["window"] = w
        metrics["context_length"] = context_length
        metrics["horizon"] = horizon

        results.append(metrics)

    return pd.DataFrame(results)


def plot_calibration_curve(df: pd.DataFrame, output_dir: Path):
    """Plot reliability diagram (calibration curve)."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Coverage vs expected
    ax = axes[0]
    expected = [50, 80, 90, 95]

    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        observed = [
            model_data[f"coverage_{e}"].mean() * 100
            for e in expected
        ]
        ax.plot(expected, observed, "o-", label=model, markersize=8)

    ax.plot([0, 100], [0, 100], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Expected Coverage (%)")
    ax.set_ylabel("Observed Coverage (%)")
    ax.set_title("Calibration: Coverage vs Expected")
    ax.legend()
    ax.set_xlim(40, 100)
    ax.set_ylim(40, 100)
    ax.grid(True, alpha=0.3)

    # Interval width comparison
    ax = axes[1]
    models = df["model"].unique()
    x = np.arange(len(models))
    width = 0.2

    for i, conf in enumerate([50, 80, 90]):
        widths = [df[df["model"] == m][f"width_{conf}"].mean() for m in models]
        ax.bar(x + i * width, widths, width, label=f"{conf}% PI")

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Normalized Interval Width")
    ax.set_title("Prediction Interval Width")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_calibration.png")
    fig.savefig(output_dir / "fig5_calibration.pdf")
    plt.close()
    print(f"Saved: fig5_calibration.png/pdf")


def plot_example_intervals(
    series: pd.Series,
    models: dict,
    output_dir: Path,
    context_length: int = 512,
    horizon: int = 24,
):
    """Plot example forecasts with prediction intervals."""

    test_start_idx = series.index.get_loc(pd.Timestamp("2023-08-01"))
    context = series.iloc[test_start_idx - context_length:test_start_idx]
    actual = series.iloc[test_start_idx:test_start_idx + horizon].values

    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]

    for ax, (model_name, model) in zip(axes, models.items()):
        point, samples = model.predict(context, prediction_length=horizon, num_samples=100)

        x = np.arange(horizon)

        # Plot actual
        ax.plot(x, actual, "k-", linewidth=2, label="Actual")

        # Plot point forecast
        ax.plot(x, point, "b-", linewidth=1.5, label="Point forecast")

        # Plot prediction intervals
        for alpha, color_alpha in [(0.5, 0.4), (0.8, 0.25), (0.9, 0.15)]:
            lower = np.quantile(samples, (1-alpha)/2, axis=0)
            upper = np.quantile(samples, 1 - (1-alpha)/2, axis=0)
            ax.fill_between(x, lower, upper, alpha=color_alpha, color="blue",
                           label=f"{int(alpha*100)}% PI")

        ax.set_ylabel("Load (MW)")
        ax.set_title(f"{model_name}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Forecast Hour")
    fig.suptitle("Probabilistic Forecasts with Prediction Intervals", y=1.02)

    plt.tight_layout()
    fig.savefig(output_dir / "fig6_prediction_intervals.png")
    fig.savefig(output_dir / "fig6_prediction_intervals.pdf")
    plt.close()
    print(f"Saved: fig6_prediction_intervals.png/pdf")


def create_calibration_table(df: pd.DataFrame, output_dir: Path):
    """Create summary table of calibration metrics."""

    summary = df.groupby("model").agg({
        "coverage_50": "mean",
        "coverage_80": "mean",
        "coverage_90": "mean",
        "coverage_95": "mean",
        "crps": "mean",
        "winkler_90": "mean",
    }).round(3)

    summary.columns = [
        "Cov. 50%", "Cov. 80%", "Cov. 90%", "Cov. 95%",
        "CRPS", "Winkler 90%"
    ]

    print("\n" + "="*70)
    print("TABLE 3: Uncertainty Calibration Metrics")
    print("="*70)
    print(summary.to_string())

    # Interpretation
    print("\n" + "-"*50)
    print("Interpretation (Perfect calibration = expected coverage)")
    print("-"*50)
    for model in summary.index:
        cov_90 = summary.loc[model, "Cov. 90%"]
        if cov_90 > 0.95:
            interp = "UNDERCONFIDENT (intervals too wide)"
        elif cov_90 < 0.85:
            interp = "OVERCONFIDENT (intervals too narrow)"
        else:
            interp = "WELL-CALIBRATED"
        print(f"{model}: {cov_90:.1%} coverage at 90% level - {interp}")

    summary.to_csv(output_dir / "table3_calibration.csv")
    print(f"\nSaved: table3_calibration.csv")


def main():
    """Run uncertainty calibration analysis."""

    output_dir = Path(__file__).parent / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    series = load_ercot_data()
    train = series[:"2022-12-31"]

    print("Loading models...")
    models = {}

    # Load Chronos-Bolt
    try:
        from energy_benchmark.models.chronos_bolt import ChronosBoltModel
        model = ChronosBoltModel(model_size="small", device="cpu")
        model.fit(train)
        models["Chronos-Bolt"] = model
        print("  Loaded Chronos-Bolt")
    except Exception as e:
        print(f"  Could not load Chronos-Bolt: {e}")

    # Load Chronos-2
    try:
        from energy_benchmark.models.chronos2 import Chronos2Model
        model = Chronos2Model(device="cpu")
        model.fit(train)
        models["Chronos-2"] = model
        print("  Loaded Chronos-2")
    except Exception as e:
        print(f"  Could not load Chronos-2: {e}")

    # Load Moirai-2
    try:
        from energy_benchmark.models.moirai import MoiraiModel
        model = MoiraiModel(model_type="moirai2", size="small", device="cpu")
        model.fit(train)
        models["Moirai-2"] = model
        print("  Loaded Moirai-2")
    except Exception as e:
        print(f"  Could not load Moirai-2: {e}")

    # Load TinyTimeMixer
    try:
        from energy_benchmark.models.tinytimemixer import TinyTimeMixerModel
        model = TinyTimeMixerModel(model_version="r2", device="cpu")
        model.fit(train)
        models["TTM"] = model
        print("  Loaded TTM")
    except Exception as e:
        print(f"  Could not load TTM: {e}")

    # Load Prophet
    try:
        from energy_benchmark.models.prophet_model import ProphetModel
        model = ProphetModel(weekly_seasonality=True, daily_seasonality=True)
        models["Prophet"] = model
        print("  Loaded Prophet")
    except Exception as e:
        print(f"  Could not load Prophet: {e}")

    if not models:
        print("No models loaded!")
        return

    # Run calibration experiment
    print("\nRunning calibration experiment...")
    all_results = []

    for model_name, model in models.items():
        print(f"  Evaluating {model_name}...")
        df = run_calibration_experiment(
            series, model_name, model,
            context_length=512,
            horizon=24,
            num_windows=30,
            num_samples=100,
        )
        all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(output_dir / "calibration_results.csv", index=False)

    # Generate outputs
    print("\nGenerating figures...")
    plot_calibration_curve(results, output_dir)
    plot_example_intervals(series, models, output_dir)
    create_calibration_table(results, output_dir)

    print("\n" + "="*70)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
