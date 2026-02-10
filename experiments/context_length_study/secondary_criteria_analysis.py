"""
Secondary Criteria Analysis for Model Selection

Since DM tests show no significant accuracy difference among top-3 foundation models
(Chronos-Bolt, Chronos-2, Moirai-2), this script evaluates secondary criteria:
1. Calibration quality (coverage at 90% nominal level)
2. Robustness (stability across test periods)
3. Inference latency
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_calibration_data(results_dir: Path) -> pd.DataFrame:
    """Load calibration results."""
    calib_file = results_dir / "figures" / "calibration_results.csv"
    if calib_file.exists():
        return pd.read_csv(calib_file)
    raise FileNotFoundError(f"Calibration file not found: {calib_file}")


def load_forecast_results(results_dir: Path) -> pd.DataFrame:
    """Load forecast results with timing and accuracy data."""
    results_file = results_dir / "raw" / "results_all_models.csv"
    if results_file.exists():
        return pd.read_csv(results_file)
    raise FileNotFoundError(f"Results file not found: {results_file}")


def analyze_calibration(calib_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze calibration quality for each model."""
    # Focus on 90% coverage level at C=512, H=24
    models_of_interest = ["Chronos-Bolt", "Chronos-2", "Moirai-2", "TTM", "Prophet"]

    results = []
    for model in models_of_interest:
        model_data = calib_df[calib_df["model"] == model]
        if len(model_data) == 0:
            continue

        # Calculate mean coverage at 90% level
        mean_coverage_90 = model_data["coverage_90"].mean()
        std_coverage_90 = model_data["coverage_90"].std()

        # Deviation from ideal (90% = 0.9)
        deviation = abs(mean_coverage_90 - 0.9)

        # CRPS (lower is better)
        mean_crps = model_data["crps"].mean()

        # Winkler score (lower is better)
        mean_winkler = model_data["winkler_90"].mean()

        # Interpretation
        if mean_coverage_90 > 0.95:
            interpretation = "Conservative (wide intervals)"
        elif mean_coverage_90 < 0.85:
            interpretation = "Overconfident (narrow intervals)"
        else:
            interpretation = "Well-calibrated"

        results.append({
            "model": model,
            "coverage_90": mean_coverage_90,
            "coverage_std": std_coverage_90,
            "deviation_from_ideal": deviation,
            "crps": mean_crps,
            "winkler_90": mean_winkler,
            "interpretation": interpretation,
        })

    return pd.DataFrame(results).sort_values("deviation_from_ideal")


def analyze_robustness(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze robustness across test periods (variance in performance)."""
    # Filter for C=512, H=24
    filtered = results_df[
        (results_df["context_length"] == 512) &
        (results_df["horizon"] == 24)
    ]

    models_of_interest = ["Chronos-Bolt", "Chronos-2", "Moirai-2", "TTM", "Prophet", "SeasonalNaive", "SARIMA"]

    results = []
    for model in models_of_interest:
        model_data = filtered[filtered["model"] == model]
        if len(model_data) == 0:
            continue

        # Calculate per-period mean MASE
        period_mase = model_data.groupby("period")["mase"].mean()

        if len(period_mase) < 2:
            continue

        # Robustness metrics
        mean_mase = period_mase.mean()
        std_mase = period_mase.std()
        cv = std_mase / mean_mase if mean_mase > 0 else float("inf")  # Coefficient of variation
        max_mase = period_mase.max()
        min_mase = period_mase.min()
        range_mase = max_mase - min_mase

        # Worst-case period
        worst_period = period_mase.idxmax()
        best_period = period_mase.idxmin()

        results.append({
            "model": model,
            "mean_mase": mean_mase,
            "std_mase": std_mase,
            "cv": cv,
            "min_mase": min_mase,
            "max_mase": max_mase,
            "range": range_mase,
            "worst_period": worst_period,
            "best_period": best_period,
        })

    return pd.DataFrame(results).sort_values("cv")


def analyze_latency(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze inference latency."""
    # Filter for C=512, H=24
    filtered = results_df[
        (results_df["context_length"] == 512) &
        (results_df["horizon"] == 24)
    ]

    models_of_interest = ["Chronos-Bolt", "Chronos-2", "Moirai-2", "TTM", "Prophet", "SeasonalNaive", "SARIMA"]

    results = []
    for model in models_of_interest:
        model_data = filtered[filtered["model"] == model]
        if len(model_data) == 0:
            continue

        mean_latency = model_data["inference_seconds"].mean()
        std_latency = model_data["inference_seconds"].std()
        max_latency = model_data["inference_seconds"].max()
        min_latency = model_data["inference_seconds"].min()

        # Convert to milliseconds for readability
        mean_ms = mean_latency * 1000

        results.append({
            "model": model,
            "mean_latency_s": mean_latency,
            "mean_latency_ms": mean_ms,
            "std_latency_s": std_latency,
            "max_latency_s": max_latency,
            "min_latency_s": min_latency,
        })

    return pd.DataFrame(results).sort_values("mean_latency_s")


def compute_composite_score(
    calib_df: pd.DataFrame,
    robust_df: pd.DataFrame,
    latency_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute composite ranking based on all criteria."""
    # Merge all metrics
    models = ["Chronos-Bolt", "Chronos-2", "Moirai-2"]

    scores = []
    for model in models:
        calib_row = calib_df[calib_df["model"] == model]
        robust_row = robust_df[robust_df["model"] == model]
        latency_row = latency_df[latency_df["model"] == model]

        if len(calib_row) == 0 or len(robust_row) == 0 or len(latency_row) == 0:
            continue

        # Extract metrics
        coverage = calib_row["coverage_90"].values[0]
        deviation = calib_row["deviation_from_ideal"].values[0]
        crps = calib_row["crps"].values[0]
        cv = robust_row["cv"].values[0]
        latency = latency_row["mean_latency_s"].values[0]

        scores.append({
            "model": model,
            "coverage_90": coverage,
            "calibration_deviation": deviation,
            "crps": crps,
            "robustness_cv": cv,
            "latency_s": latency,
        })

    df = pd.DataFrame(scores)

    # Rank each metric (lower rank is better)
    df["rank_calibration"] = df["calibration_deviation"].rank()
    df["rank_crps"] = df["crps"].rank()
    df["rank_robustness"] = df["robustness_cv"].rank()
    df["rank_latency"] = df["latency_s"].rank()

    # Composite score (sum of ranks, lower is better)
    df["composite_rank"] = (
        df["rank_calibration"] +
        df["rank_crps"] +
        df["rank_robustness"] +
        df["rank_latency"]
    )

    return df.sort_values("composite_rank")


def main():
    results_dir = Path("results")
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SECONDARY CRITERIA ANALYSIS FOR MODEL SELECTION")
    print("=" * 80)
    print("\nSince DM tests show no significant accuracy difference among")
    print("Chronos-Bolt, Chronos-2, and Moirai-2, we evaluate secondary criteria.\n")

    # Load data
    try:
        calib_df = load_calibration_data(results_dir)
        logger.info(f"Loaded {len(calib_df)} calibration records")
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    try:
        results_df = load_forecast_results(results_dir)
        logger.info(f"Loaded {len(results_df)} forecast results")
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # =========================================================================
    # 1. Calibration Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. CALIBRATION QUALITY (90% Prediction Intervals)")
    print("=" * 80)
    print("Target: 90% coverage. Deviation from 90% indicates miscalibration.\n")

    calib_analysis = analyze_calibration(calib_df)

    print(f"{'Model':<15} {'Coverage':<12} {'Deviation':<12} {'CRPS':<10} {'Interpretation':<25}")
    print("-" * 74)
    for _, row in calib_analysis.iterrows():
        print(f"{row['model']:<15} {row['coverage_90']:.1%}       {row['deviation_from_ideal']:.3f}        {row['crps']:.0f}       {row['interpretation']}")

    # Find winner
    top3_calib = calib_analysis[calib_analysis["model"].isin(["Chronos-Bolt", "Chronos-2", "Moirai-2"])]
    calib_winner = top3_calib.iloc[0]["model"]
    print(f"\n>>> Best calibrated (among top-3): {calib_winner}")

    # =========================================================================
    # 2. Robustness Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. ROBUSTNESS (Stability Across Test Periods)")
    print("=" * 80)
    print("Lower CV (coefficient of variation) = more stable performance.\n")

    robust_analysis = analyze_robustness(results_df)

    print(f"{'Model':<15} {'Mean MASE':<12} {'Std':<10} {'CV':<10} {'Worst Period':<15}")
    print("-" * 62)
    for _, row in robust_analysis.iterrows():
        print(f"{row['model']:<15} {row['mean_mase']:.3f}        {row['std_mase']:.3f}      {row['cv']:.3f}     {row['worst_period']}")

    # Find winner
    top3_robust = robust_analysis[robust_analysis["model"].isin(["Chronos-Bolt", "Chronos-2", "Moirai-2"])]
    robust_winner = top3_robust.iloc[0]["model"]
    print(f"\n>>> Most robust (among top-3): {robust_winner}")

    # =========================================================================
    # 3. Latency Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. INFERENCE LATENCY")
    print("=" * 80)
    print("Lower latency = faster inference, better for real-time applications.\n")

    latency_analysis = analyze_latency(results_df)

    print(f"{'Model':<15} {'Mean (s)':<12} {'Mean (ms)':<12} {'Speedup vs Slowest':<20}")
    print("-" * 59)
    max_latency = latency_analysis["mean_latency_s"].max()
    for _, row in latency_analysis.iterrows():
        speedup = max_latency / row["mean_latency_s"] if row["mean_latency_s"] > 0 else float("inf")
        print(f"{row['model']:<15} {row['mean_latency_s']:.4f}       {row['mean_latency_ms']:.1f}         {speedup:.1f}x")

    # Find winner
    top3_latency = latency_analysis[latency_analysis["model"].isin(["Chronos-Bolt", "Chronos-2", "Moirai-2"])]
    latency_winner = top3_latency.iloc[0]["model"]
    print(f"\n>>> Fastest (among top-3): {latency_winner}")

    # =========================================================================
    # 4. Composite Ranking
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. COMPOSITE RANKING (All Criteria Combined)")
    print("=" * 80)
    print("Ranks: Calibration + CRPS + Robustness + Latency (lower total = better)\n")

    composite = compute_composite_score(calib_analysis, robust_analysis, latency_analysis)

    print(f"{'Model':<15} {'Calib Rank':<12} {'CRPS Rank':<12} {'Robust Rank':<13} {'Latency Rank':<13} {'TOTAL':<8}")
    print("-" * 73)
    for _, row in composite.iterrows():
        print(f"{row['model']:<15} {int(row['rank_calibration']):<12} {int(row['rank_crps']):<12} {int(row['rank_robustness']):<13} {int(row['rank_latency']):<13} {row['composite_rank']:.0f}")

    overall_winner = composite.iloc[0]["model"]

    # =========================================================================
    # 5. Final Recommendation
    # =========================================================================
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    print(f"""
Based on secondary criteria analysis:

  - Best Calibration:  {calib_winner}
  - Most Robust:       {robust_winner}
  - Fastest:           {latency_winner}

  >>> OVERALL WINNER: {overall_winner}

Detailed findings:
""")

    # Generate detailed interpretation
    winner_row = composite[composite["model"] == overall_winner].iloc[0]
    winner_calib = calib_analysis[calib_analysis["model"] == overall_winner].iloc[0]
    winner_robust = robust_analysis[robust_analysis["model"] == overall_winner].iloc[0]
    winner_latency = latency_analysis[latency_analysis["model"] == overall_winner].iloc[0]

    interpretation = f"""
1. CALIBRATION: {overall_winner} achieves {winner_calib['coverage_90']:.1%} empirical coverage
   at the 90% nominal level ({winner_calib['interpretation'].lower()}).

2. ROBUSTNESS: Coefficient of variation = {winner_robust['cv']:.3f}, meaning performance
   remains stable across summer heat waves, winter peaks, and COVID anomalies.

3. LATENCY: Mean inference time of {winner_latency['mean_latency_s']:.3f}s per 24-hour forecast,
   suitable for real-time operational deployment.

Since all three models show statistically equivalent point accuracy (DM test p > 0.05),
{overall_winner} is recommended as the preferred choice for ERCOT load forecasting.
"""
    print(interpretation)

    # Save results
    composite.to_csv(output_dir / "secondary_criteria_ranking.csv", index=False)
    calib_analysis.to_csv(output_dir / "calibration_analysis.csv", index=False)
    robust_analysis.to_csv(output_dir / "robustness_analysis.csv", index=False)
    latency_analysis.to_csv(output_dir / "latency_analysis.csv", index=False)

    print(f"\nResults saved to: {output_dir}")

    # Return for LaTeX integration
    return {
        "winner": overall_winner,
        "calibration": calib_analysis,
        "robustness": robust_analysis,
        "latency": latency_analysis,
        "composite": composite,
    }


if __name__ == "__main__":
    results = main()
