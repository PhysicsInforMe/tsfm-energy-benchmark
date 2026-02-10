"""
Diebold-Mariano Statistical Tests for TSFM Paper

Computes pairwise DM tests between models at C=512, H=24
and outputs results to fill LaTeX placeholders.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Diebold-Mariano Test Implementation
# =============================================================================

def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 1):
    """
    Diebold-Mariano test for equal predictive accuracy.

    Parameters:
    -----------
    e1 : array-like — Forecast errors from model 1
    e2 : array-like — Forecast errors from model 2
    h : int — Forecast horizon (for HAC variance correction)
    power : int — 1 for absolute errors, 2 for squared errors

    Returns:
    --------
    dm_stat : float — DM test statistic
    p_value : float — Two-sided p-value
    """
    e1 = np.asarray(e1).flatten()
    e2 = np.asarray(e2).flatten()

    # Loss differential
    d = np.abs(e1)**power - np.abs(e2)**power
    n = len(d)
    d_mean = np.mean(d)

    # Newey-West HAC variance estimate
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, h):
        if k < n:
            gamma_k = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
            gamma_sum += 2 * gamma_k

    var_d = (gamma_0 + gamma_sum) / n

    if var_d <= 0:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=n-1)

    return dm_stat, p_value


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load ERCOT data."""
    from energy_benchmark.data import ERCOTLoader
    from energy_benchmark.data.preprocessing import preprocess_series

    logger.info("Loading ERCOT data...")
    loader = ERCOTLoader(years=[2020, 2021, 2022, 2023, 2024])
    series = loader.load()
    series = preprocess_series(series)
    logger.info(f"Loaded {len(series):,} observations")
    return series


def create_model(model_type: str, config: dict):
    """Create model instance."""
    if model_type == "seasonal_naive":
        from energy_benchmark.models import SeasonalNaiveModel
        return SeasonalNaiveModel(**config)
    elif model_type == "chronos_bolt":
        from energy_benchmark.models.chronos_bolt import ChronosBoltModel
        return ChronosBoltModel(**config)
    elif model_type == "chronos2":
        from energy_benchmark.models.chronos2 import Chronos2Model
        return Chronos2Model(**config)
    elif model_type == "moirai":
        from energy_benchmark.models.moirai import MoiraiModel
        return MoiraiModel(**config)
    else:
        raise ValueError(f"Unknown model: {model_type}")


# =============================================================================
# Collect Per-Timestep Errors
# =============================================================================

def collect_errors_for_model(
    model,
    model_name: str,
    series: pd.Series,
    context_length: int = 512,
    horizon: int = 24,
    num_windows: int = 10,
    window_step: int = 24,
) -> np.ndarray:
    """
    Collect per-timestep forecast errors for a model.

    Returns array of shape (num_total_timesteps,) with all errors concatenated.
    """
    test_periods = [
        ("summer_2023", "2023-07-01", "2023-08-31"),
        ("winter_2022", "2022-12-01", "2023-02-28"),
        ("covid_2020", "2020-03-15", "2020-04-30"),
    ]

    all_errors = []

    for period_name, period_start, period_end in test_periods:
        period_data = series[period_start:period_end]
        if len(period_data) < horizon + context_length:
            continue

        period_start_ts = pd.Timestamp(period_start)
        earliest_forecast = series.index.get_loc(period_start_ts)

        if earliest_forecast < context_length:
            continue

        for w in range(num_windows):
            forecast_offset = w * window_step
            if forecast_offset + horizon > len(period_data):
                break

            forecast_start_idx = earliest_forecast + forecast_offset
            ctx_start_idx = forecast_start_idx - context_length
            context = series.iloc[ctx_start_idx:forecast_start_idx]
            actual = series.iloc[forecast_start_idx:forecast_start_idx + horizon].values

            if len(actual) < horizon:
                break

            try:
                point_forecast, _ = model.predict(
                    context, prediction_length=horizon, num_samples=50
                )
                # Per-timestep errors
                errors = actual - point_forecast
                all_errors.extend(errors.tolist())
            except Exception as e:
                logger.warning(f"{model_name} prediction failed: {e}")
                continue

    return np.array(all_errors)


# =============================================================================
# Main
# =============================================================================

def main():
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    series = load_data()

    # Models to test
    model_configs = [
        ("Chronos-Bolt", "chronos_bolt", {"model_size": "small", "device": "cpu"}),
        ("Chronos-2", "chronos2", {"device": "cpu"}),
        ("Moirai-2", "moirai", {"model_type": "moirai2", "size": "small", "device": "cpu"}),
        ("SeasonalNaive", "seasonal_naive", {"seasonality": 168}),
    ]

    # Collect errors for each model
    errors = {}

    for name, model_type, config in model_configs:
        logger.info(f"Collecting errors for {name}...")
        model = create_model(model_type, config)

        # Fit if needed
        if hasattr(model, 'fit'):
            train = series[:"2022-12-31"]
            model.fit(train)

        errors[name] = collect_errors_for_model(
            model, name, series,
            context_length=512,
            horizon=24,
            num_windows=10,
            window_step=24,
        )
        logger.info(f"  Collected {len(errors[name])} error values")

    # Ensure all error arrays have same length
    min_len = min(len(e) for e in errors.values())
    for name in errors:
        errors[name] = errors[name][:min_len]

    logger.info(f"\nUsing {min_len} aligned error values for DM tests")

    # ==========================================================================
    # Run DM Tests
    # ==========================================================================

    print("\n" + "="*70)
    print("DIEBOLD-MARIANO TEST RESULTS")
    print("="*70)
    print(f"Context: 512h, Horizon: 24h, Sample size: n={min_len}")
    print("="*70 + "\n")

    results = {}

    # 1. Chronos-Bolt vs Chronos-2
    dm_stat, p_val = diebold_mariano_test(
        errors["Chronos-Bolt"], errors["Chronos-2"], h=1, power=1
    )
    results["BOLT_VS_CHR2"] = (dm_stat, p_val)
    print(f"Chronos-Bolt vs Chronos-2:")
    print(f"  DM statistic: {dm_stat:.3f}")
    print(f"  p-value: {p_val:.4f}")
    print(f"  Significant at alpha=0.05: {'Yes' if p_val < 0.05 else 'No'}\n")

    # 2. Chronos-Bolt vs Moirai-2
    dm_stat, p_val = diebold_mariano_test(
        errors["Chronos-Bolt"], errors["Moirai-2"], h=1, power=1
    )
    results["BOLT_VS_MOIRAI"] = (dm_stat, p_val)
    print(f"Chronos-Bolt vs Moirai-2:")
    print(f"  DM statistic: {dm_stat:.3f}")
    print(f"  p-value: {p_val:.4f}")
    print(f"  Significant at alpha=0.05: {'Yes' if p_val < 0.05 else 'No'}\n")

    # 3. Chronos-2 vs Moirai-2
    dm_stat, p_val = diebold_mariano_test(
        errors["Chronos-2"], errors["Moirai-2"], h=1, power=1
    )
    results["CHR2_VS_MOIRAI"] = (dm_stat, p_val)
    print(f"Chronos-2 vs Moirai-2:")
    print(f"  DM statistic: {dm_stat:.3f}")
    print(f"  p-value: {p_val:.4f}")
    print(f"  Significant at alpha=0.05: {'Yes' if p_val < 0.05 else 'No'}\n")

    # 4-6. Foundation models vs Seasonal Naive
    fm_vs_naive_pvals = []
    for fm_name in ["Chronos-Bolt", "Chronos-2", "Moirai-2"]:
        dm_stat, p_val = diebold_mariano_test(
            errors[fm_name], errors["SeasonalNaive"], h=1, power=1
        )
        fm_vs_naive_pvals.append(p_val)
        print(f"{fm_name} vs Seasonal Naive:")
        print(f"  DM statistic: {dm_stat:.3f}")
        print(f"  p-value: {p_val:.4f}")
        print(f"  Significant at alpha=0.05: {'Yes' if p_val < 0.05 else 'No'}\n")

    max_naive_pval = max(fm_vs_naive_pvals)
    results["FM_VS_NAIVE_P"] = max_naive_pval

    # ==========================================================================
    # Generate LaTeX replacement strings
    # ==========================================================================

    print("="*70)
    print("LATEX PLACEHOLDER VALUES")
    print("="*70 + "\n")

    print("Copy these values to replace placeholders in arxiv_paper_v2.tex:\n")

    print(f"[DM_BOLT_VS_CHR2_STAT] -> ${results['BOLT_VS_CHR2'][0]:.3f}$")
    print(f"[DM_BOLT_VS_CHR2_P] -> ${results['BOLT_VS_CHR2'][1]:.4f}$")
    print(f"[DM_BOLT_VS_MOIRAI_STAT] -> ${results['BOLT_VS_MOIRAI'][0]:.3f}$")
    print(f"[DM_BOLT_VS_MOIRAI_P] -> ${results['BOLT_VS_MOIRAI'][1]:.4f}$")
    print(f"[DM_CHR2_VS_MOIRAI_STAT] -> ${results['CHR2_VS_MOIRAI'][0]:.3f}$")
    print(f"[DM_CHR2_VS_MOIRAI_P] -> ${results['CHR2_VS_MOIRAI'][1]:.4f}$")
    print(f"[DM_FM_VS_NAIVE_P] -> ${max_naive_pval:.4f}$")

    # Interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)

    top3_pvals = [
        results["BOLT_VS_CHR2"][1],
        results["BOLT_VS_MOIRAI"][1],
        results["CHR2_VS_MOIRAI"][1],
    ]

    any_significant = any(p < 0.05 for p in top3_pvals)

    if not any_significant:
        interpretation = (
            "The differences among the top three foundation models are not "
            "statistically significant at the 5\\% level ($p > 0.05$ for all "
            "pairwise comparisons), suggesting that model selection within this "
            "cluster should be guided by secondary criteria---calibration quality, "
            "robustness to distribution shift, and inference latency---rather than "
            "point accuracy alone."
        )
    else:
        # Find which comparisons are significant
        sig_pairs = []
        if results["BOLT_VS_CHR2"][1] < 0.05:
            winner = "Chronos-Bolt" if results["BOLT_VS_CHR2"][0] < 0 else "Chronos-2"
            sig_pairs.append(f"{winner} significantly outperforms the other Chronos variant ($p = {results['BOLT_VS_CHR2'][1]:.4f}$)")
        if results["BOLT_VS_MOIRAI"][1] < 0.05:
            winner = "Chronos-Bolt" if results["BOLT_VS_MOIRAI"][0] < 0 else "Moirai-2"
            sig_pairs.append(f"{winner} significantly outperforms the other ($p = {results['BOLT_VS_MOIRAI'][1]:.4f}$)")
        if results["CHR2_VS_MOIRAI"][1] < 0.05:
            winner = "Chronos-2" if results["CHR2_VS_MOIRAI"][0] < 0 else "Moirai-2"
            sig_pairs.append(f"{winner} significantly outperforms the other ($p = {results['CHR2_VS_MOIRAI'][1]:.4f}$)")

        interpretation = "The DM test reveals that " + "; ".join(sig_pairs) + "."

    print(f"\n{interpretation}\n")

    # Save results
    results_df = pd.DataFrame([
        {"comparison": "Bolt vs Chr2", "dm_stat": results["BOLT_VS_CHR2"][0], "p_value": results["BOLT_VS_CHR2"][1]},
        {"comparison": "Bolt vs Moirai", "dm_stat": results["BOLT_VS_MOIRAI"][0], "p_value": results["BOLT_VS_MOIRAI"][1]},
        {"comparison": "Chr2 vs Moirai", "dm_stat": results["CHR2_VS_MOIRAI"][0], "p_value": results["CHR2_VS_MOIRAI"][1]},
        {"comparison": "FM vs Naive (max p)", "dm_stat": None, "p_value": max_naive_pval},
    ])
    results_df.to_csv(output_dir / "dm_test_results.csv", index=False)
    print(f"Results saved to: {output_dir / 'dm_test_results.csv'}")

    return results, interpretation


if __name__ == "__main__":
    results, interpretation = main()
