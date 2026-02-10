"""
Prescriptive Analysis for Energy Load Forecasting

This module goes beyond prediction to provide actionable recommendations:
1. Peak Load Detection & Demand Response
2. Optimal Reserve Margin Planning
3. Cost Optimization under Uncertainty
4. Risk-Aware Decision Making

These analyses translate probabilistic forecasts into operational decisions.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


# =============================================================================
# Peak Load Detection & Demand Response
# =============================================================================

def detect_peak_events(
    samples: np.ndarray,
    threshold_percentile: float = 95.0,
    historical_data: Optional[np.ndarray] = None,
) -> dict:
    """
    Detect potential peak load events from probabilistic forecasts.

    Parameters
    ----------
    samples : np.ndarray
        Probabilistic samples of shape (num_samples, horizon)
    threshold_percentile : float
        Percentile of historical data to define peak threshold
    historical_data : np.ndarray, optional
        Historical data for computing threshold

    Returns
    -------
    dict with peak detection results
    """
    # Compute statistics from samples
    median_forecast = np.median(samples, axis=0)
    upper_95 = np.quantile(samples, 0.95, axis=0)
    upper_99 = np.quantile(samples, 0.99, axis=0)

    # Define peak threshold
    if historical_data is not None:
        peak_threshold = np.percentile(historical_data, threshold_percentile)
    else:
        # Use a reasonable default based on forecast
        peak_threshold = np.percentile(samples, threshold_percentile)

    # Probability of exceeding threshold at each timestep
    prob_exceed = np.mean(samples > peak_threshold, axis=0)

    # Risk score (weighted by exceedance amount)
    exceedance = np.maximum(samples - peak_threshold, 0)
    expected_exceedance = np.mean(exceedance, axis=0)

    # Peak event detection
    peak_hours = np.where(prob_exceed > 0.5)[0]  # >50% chance of peak

    return {
        "threshold": peak_threshold,
        "prob_exceed": prob_exceed,
        "expected_exceedance": expected_exceedance,
        "peak_hours": peak_hours,
        "median_forecast": median_forecast,
        "upper_95": upper_95,
        "upper_99": upper_99,
        "max_prob": prob_exceed.max(),
        "max_expected_exceedance": expected_exceedance.max(),
    }


def recommend_demand_response(
    peak_results: dict,
    capacity_mw: float = 5000.0,
    dr_cost_per_mw: float = 100.0,
    peak_penalty_per_mw: float = 500.0,
) -> dict:
    """
    Recommend demand response actions based on peak forecasts.

    Parameters
    ----------
    peak_results : dict
        Results from detect_peak_events()
    capacity_mw : float
        Available demand response capacity in MW
    dr_cost_per_mw : float
        Cost per MW of demand response activation
    peak_penalty_per_mw : float
        Penalty per MW of peak exceedance

    Returns
    -------
    dict with demand response recommendations
    """
    prob_exceed = peak_results["prob_exceed"]
    expected_exceedance = peak_results["expected_exceedance"]

    # Optimal DR decision at each hour (risk-neutral)
    # Activate DR if: P(peak) * penalty > DR_cost
    dr_threshold = dr_cost_per_mw / peak_penalty_per_mw

    recommended_dr = np.zeros_like(prob_exceed)
    for t in range(len(prob_exceed)):
        if prob_exceed[t] > dr_threshold:
            # Amount of DR to activate: min of expected exceedance and capacity
            recommended_dr[t] = min(expected_exceedance[t], capacity_mw)

    # Calculate expected costs
    # Without DR: expected penalty
    expected_penalty_no_dr = expected_exceedance * peak_penalty_per_mw

    # With DR: DR cost + residual penalty
    residual_exceedance = np.maximum(expected_exceedance - recommended_dr, 0)
    expected_cost_with_dr = (
        recommended_dr * dr_cost_per_mw +
        residual_exceedance * peak_penalty_per_mw
    )

    total_savings = expected_penalty_no_dr.sum() - expected_cost_with_dr.sum()

    return {
        "recommended_dr_mw": recommended_dr,
        "dr_activation_hours": np.where(recommended_dr > 0)[0],
        "total_dr_capacity_used": recommended_dr.sum(),
        "expected_cost_no_dr": expected_penalty_no_dr.sum(),
        "expected_cost_with_dr": expected_cost_with_dr.sum(),
        "expected_savings": total_savings,
        "dr_threshold_prob": dr_threshold,
    }


# =============================================================================
# Reserve Margin Planning
# =============================================================================

def calculate_reserve_requirements(
    samples: np.ndarray,
    reliability_target: float = 0.999,  # 99.9% reliability
    fixed_reserve_pct: float = 0.10,    # 10% fixed reserve
) -> dict:
    """
    Calculate spinning reserve requirements based on forecast uncertainty.

    Parameters
    ----------
    samples : np.ndarray
        Probabilistic samples of shape (num_samples, horizon)
    reliability_target : float
        Target probability of meeting demand
    fixed_reserve_pct : float
        Fixed reserve percentage (traditional approach)

    Returns
    -------
    dict with reserve requirement analysis
    """
    median_forecast = np.median(samples, axis=0)

    # Traditional fixed reserve (percentage of median)
    fixed_reserve = median_forecast * fixed_reserve_pct

    # Probabilistic reserve: quantile-based
    # Reserve = Q(reliability) - median
    upper_quantile = np.quantile(samples, reliability_target, axis=0)
    probabilistic_reserve = upper_quantile - median_forecast

    # Calculate reserve savings (positive = savings with probabilistic approach)
    reserve_difference = fixed_reserve - probabilistic_reserve

    # Hours where probabilistic reserve is lower (savings opportunity)
    savings_hours = np.where(reserve_difference > 0)[0]

    return {
        "median_forecast": median_forecast,
        "fixed_reserve": fixed_reserve,
        "probabilistic_reserve": probabilistic_reserve,
        "reserve_difference": reserve_difference,
        "total_fixed_reserve_mwh": fixed_reserve.sum(),
        "total_prob_reserve_mwh": probabilistic_reserve.sum(),
        "savings_hours": savings_hours,
        "avg_savings_pct": np.mean(reserve_difference / fixed_reserve) * 100,
        "reliability_target": reliability_target,
    }


# =============================================================================
# Cost Optimization under Uncertainty
# =============================================================================

def energy_cost_optimization(
    samples: np.ndarray,
    spot_prices: np.ndarray,
    storage_capacity_mwh: float = 1000.0,
    storage_efficiency: float = 0.85,
    max_charge_rate: float = 250.0,
) -> dict:
    """
    Optimize energy storage dispatch under forecast uncertainty.

    Parameters
    ----------
    samples : np.ndarray
        Probabilistic load samples of shape (num_samples, horizon)
    spot_prices : np.ndarray
        Spot electricity prices ($/MWh) for each hour
    storage_capacity_mwh : float
        Battery storage capacity in MWh
    storage_efficiency : float
        Round-trip efficiency of storage
    max_charge_rate : float
        Maximum charge/discharge rate in MW

    Returns
    -------
    dict with optimization results
    """
    horizon = samples.shape[1]
    median_forecast = np.median(samples, axis=0)

    # Simple heuristic: charge during low-price hours, discharge during high-price
    price_percentile = np.zeros(horizon)
    for t in range(horizon):
        price_percentile[t] = np.mean(spot_prices[:t+1] <= spot_prices[t]) if t > 0 else 0.5

    # Storage state
    storage_level = np.zeros(horizon + 1)
    storage_level[0] = storage_capacity_mwh * 0.5  # Start at 50%

    charge_schedule = np.zeros(horizon)
    discharge_schedule = np.zeros(horizon)

    for t in range(horizon):
        if price_percentile[t] < 0.3:  # Low price - charge
            charge = min(
                max_charge_rate,
                storage_capacity_mwh - storage_level[t]
            )
            charge_schedule[t] = charge
            storage_level[t + 1] = storage_level[t] + charge * storage_efficiency
        elif price_percentile[t] > 0.7:  # High price - discharge
            discharge = min(
                max_charge_rate,
                storage_level[t]
            )
            discharge_schedule[t] = discharge
            storage_level[t + 1] = storage_level[t] - discharge
        else:
            storage_level[t + 1] = storage_level[t]

    # Calculate cost impact
    charging_cost = np.sum(charge_schedule * spot_prices)
    discharge_revenue = np.sum(discharge_schedule * spot_prices)
    net_benefit = discharge_revenue - charging_cost

    # Risk-adjusted benefit considering forecast uncertainty
    # Higher uncertainty = more conservative strategy needed
    forecast_std = np.std(samples, axis=0)
    risk_factor = np.mean(forecast_std / median_forecast)

    return {
        "charge_schedule": charge_schedule,
        "discharge_schedule": discharge_schedule,
        "storage_level": storage_level,
        "charging_cost": charging_cost,
        "discharge_revenue": discharge_revenue,
        "net_benefit": net_benefit,
        "risk_factor": risk_factor,
        "total_energy_charged": charge_schedule.sum(),
        "total_energy_discharged": discharge_schedule.sum(),
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_peak_detection(
    context: np.ndarray,
    samples: np.ndarray,
    peak_results: dict,
    dr_results: dict,
    output_path: Path,
):
    """Create peak detection visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    horizon = samples.shape[1]
    x = np.arange(horizon)

    # Top: Forecast with uncertainty
    ax = axes[0]
    median = peak_results["median_forecast"]
    upper_95 = peak_results["upper_95"]
    lower_95 = np.quantile(samples, 0.05, axis=0)

    ax.fill_between(x, lower_95, upper_95, alpha=0.3, color="blue", label="90% PI")
    ax.plot(x, median, "b-", linewidth=2, label="Median Forecast")
    ax.axhline(peak_results["threshold"], color="red", linestyle="--",
               linewidth=1.5, label=f"Peak Threshold ({peak_results['threshold']:.0f} MW)")

    # Highlight peak hours
    for hour in peak_results["peak_hours"]:
        ax.axvspan(hour - 0.5, hour + 0.5, alpha=0.2, color="red")

    ax.set_ylabel("Load (MW)")
    ax.set_title("Load Forecast with Peak Detection")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Middle: Peak probability
    ax = axes[1]
    ax.bar(x, peak_results["prob_exceed"], color="orange", alpha=0.7)
    ax.axhline(dr_results["dr_threshold_prob"], color="green", linestyle="--",
               label=f"DR Activation Threshold ({dr_results['dr_threshold_prob']:.0%})")
    ax.set_ylabel("P(Peak)")
    ax.set_title("Probability of Exceeding Peak Threshold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Bottom: Demand Response Recommendation
    ax = axes[2]
    ax.bar(x, dr_results["recommended_dr_mw"], color="green", alpha=0.7,
           label="Recommended DR")
    ax.set_ylabel("DR (MW)")
    ax.set_xlabel("Forecast Hour")
    ax.set_title(f"Demand Response Recommendations (Expected Savings: ${dr_results['expected_savings']:,.0f})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path / "fig7_peak_detection.png")
    fig.savefig(output_path / "fig7_peak_detection.pdf")
    plt.close()
    print(f"Saved: fig7_peak_detection.png/pdf")


def plot_reserve_analysis(
    reserve_results: dict,
    output_path: Path,
):
    """Create reserve margin analysis visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    horizon = len(reserve_results["median_forecast"])
    x = np.arange(horizon)

    # Top: Reserve comparison
    ax = axes[0]
    ax.plot(x, reserve_results["fixed_reserve"], "r--", linewidth=2,
            label=f"Fixed 10% Reserve")
    ax.plot(x, reserve_results["probabilistic_reserve"], "g-", linewidth=2,
            label=f"Probabilistic Reserve (99.9% reliability)")
    ax.fill_between(x, 0, reserve_results["fixed_reserve"], alpha=0.2, color="red")
    ax.fill_between(x, 0, reserve_results["probabilistic_reserve"], alpha=0.2, color="green")

    ax.set_ylabel("Reserve (MW)")
    ax.set_title("Reserve Margin: Fixed vs Probabilistic Approach")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Bottom: Savings
    ax = axes[1]
    savings = reserve_results["reserve_difference"]
    colors = ["green" if s > 0 else "red" for s in savings]
    ax.bar(x, savings, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Reserve Savings (MW)")
    ax.set_xlabel("Forecast Hour")
    ax.set_title(f"Potential Savings with Probabilistic Reserves (Avg: {reserve_results['avg_savings_pct']:.1f}%)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path / "fig8_reserve_analysis.png")
    fig.savefig(output_path / "fig8_reserve_analysis.pdf")
    plt.close()
    print(f"Saved: fig8_reserve_analysis.png/pdf")


def plot_storage_optimization(
    optimization_results: dict,
    spot_prices: np.ndarray,
    output_path: Path,
):
    """Create storage optimization visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    horizon = len(spot_prices)
    x = np.arange(horizon)

    # Top: Spot prices
    ax = axes[0]
    ax.plot(x, spot_prices, "b-", linewidth=1.5)
    ax.fill_between(x, 0, spot_prices, alpha=0.3, color="blue")
    ax.set_ylabel("Price ($/MWh)")
    ax.set_title("Electricity Spot Prices")
    ax.grid(True, alpha=0.3)

    # Middle: Charge/Discharge schedule
    ax = axes[1]
    ax.bar(x, optimization_results["charge_schedule"], color="green", alpha=0.7,
           label="Charging")
    ax.bar(x, -optimization_results["discharge_schedule"], color="red", alpha=0.7,
           label="Discharging")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Power (MW)")
    ax.set_title("Optimal Storage Dispatch Schedule")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Bottom: Storage state of charge
    ax = axes[2]
    storage = optimization_results["storage_level"][:-1]  # Exclude final state
    ax.fill_between(x, 0, storage, alpha=0.5, color="purple")
    ax.plot(x, storage, "purple", linewidth=2)
    ax.set_ylabel("Energy (MWh)")
    ax.set_xlabel("Hour")
    ax.set_title(f"Battery State of Charge (Net Benefit: ${optimization_results['net_benefit']:,.0f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path / "fig9_storage_optimization.png")
    fig.savefig(output_path / "fig9_storage_optimization.pdf")
    plt.close()
    print(f"Saved: fig9_storage_optimization.png/pdf")


def create_prescriptive_summary(
    peak_results: dict,
    dr_results: dict,
    reserve_results: dict,
    optimization_results: dict,
    output_path: Path,
):
    """Create summary table of prescriptive analysis results."""
    summary = {
        "Peak Detection": {
            "Peak Threshold (MW)": f"{peak_results['threshold']:,.0f}",
            "Max Peak Probability": f"{peak_results['max_prob']:.1%}",
            "Peak Hours Detected": len(peak_results["peak_hours"]),
            "Max Expected Exceedance (MW)": f"{peak_results['max_expected_exceedance']:,.0f}",
        },
        "Demand Response": {
            "DR Hours Recommended": len(dr_results["dr_activation_hours"]),
            "Total DR Capacity Used (MWh)": f"{dr_results['total_dr_capacity_used']:,.0f}",
            "Expected Cost without DR ($)": f"{dr_results['expected_cost_no_dr']:,.0f}",
            "Expected Cost with DR ($)": f"{dr_results['expected_cost_with_dr']:,.0f}",
            "Expected Savings ($)": f"{dr_results['expected_savings']:,.0f}",
        },
        "Reserve Planning": {
            "Total Fixed Reserve (MWh)": f"{reserve_results['total_fixed_reserve_mwh']:,.0f}",
            "Total Probabilistic Reserve (MWh)": f"{reserve_results['total_prob_reserve_mwh']:,.0f}",
            "Average Savings (%)": f"{reserve_results['avg_savings_pct']:.1f}%",
            "Reliability Target": f"{reserve_results['reliability_target']:.1%}",
        },
        "Storage Optimization": {
            "Energy Charged (MWh)": f"{optimization_results['total_energy_charged']:,.0f}",
            "Energy Discharged (MWh)": f"{optimization_results['total_energy_discharged']:,.0f}",
            "Net Benefit ($)": f"{optimization_results['net_benefit']:,.0f}",
            "Forecast Risk Factor": f"{optimization_results['risk_factor']:.3f}",
        },
    }

    print("\n" + "=" * 70)
    print("PRESCRIPTIVE ANALYSIS SUMMARY")
    print("=" * 70)

    for section, metrics in summary.items():
        print(f"\n{section}:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    # Save to CSV
    rows = []
    for section, metrics in summary.items():
        for metric, value in metrics.items():
            rows.append({"Section": section, "Metric": metric, "Value": value})
    df = pd.DataFrame(rows)
    df.to_csv(output_path / "table4_prescriptive_summary.csv", index=False)
    print(f"\nSaved: table4_prescriptive_summary.csv")


def main():
    """Run prescriptive analysis."""
    output_dir = Path(__file__).parent / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    series = load_ercot_data()
    train = series[:"2022-12-31"]

    # Use a high-demand summer period for prescriptive analysis
    test_start = pd.Timestamp("2023-07-15")
    context_length = 512
    horizon = 24

    test_start_idx = series.index.get_loc(test_start)
    context = series.iloc[test_start_idx - context_length:test_start_idx]

    print("Loading best model for prescriptive analysis...")
    # Use Chronos-2 as the primary model (or Moirai-2)
    try:
        from energy_benchmark.models.moirai import MoiraiModel
        model = MoiraiModel(model_type="moirai2", size="small", device="cpu")
        model_name = "Moirai-2"
    except Exception:
        from energy_benchmark.models.chronos2 import Chronos2Model
        model = Chronos2Model(device="cpu")
        model_name = "Chronos-2"

    model.fit(train)
    print(f"  Using {model_name}")

    print("\nGenerating probabilistic forecast...")
    point_forecast, samples = model.predict(
        context,
        prediction_length=horizon,
        num_samples=500,  # More samples for better uncertainty estimation
    )

    if samples is None:
        print("ERROR: Model did not return probabilistic samples")
        return

    print(f"  Forecast shape: {point_forecast.shape}, Samples shape: {samples.shape}")

    # Historical data for peak threshold
    historical = series[:"2023-07-01"].values

    # ==== Peak Detection ====
    print("\n1. Running Peak Detection Analysis...")
    peak_results = detect_peak_events(
        samples,
        threshold_percentile=95.0,
        historical_data=historical,
    )
    print(f"   Peak threshold: {peak_results['threshold']:,.0f} MW")
    print(f"   Peak hours detected: {len(peak_results['peak_hours'])}")

    # ==== Demand Response ====
    print("\n2. Computing Demand Response Recommendations...")
    dr_results = recommend_demand_response(
        peak_results,
        capacity_mw=5000.0,
        dr_cost_per_mw=100.0,
        peak_penalty_per_mw=500.0,
    )
    print(f"   DR activation hours: {len(dr_results['dr_activation_hours'])}")
    print(f"   Expected savings: ${dr_results['expected_savings']:,.0f}")

    # ==== Reserve Planning ====
    print("\n3. Calculating Reserve Requirements...")
    reserve_results = calculate_reserve_requirements(
        samples,
        reliability_target=0.999,
        fixed_reserve_pct=0.10,
    )
    print(f"   Fixed reserve: {reserve_results['total_fixed_reserve_mwh']:,.0f} MWh")
    print(f"   Probabilistic reserve: {reserve_results['total_prob_reserve_mwh']:,.0f} MWh")
    print(f"   Average savings: {reserve_results['avg_savings_pct']:.1f}%")

    # ==== Storage Optimization ====
    print("\n4. Optimizing Energy Storage...")
    # Generate synthetic spot prices (realistic pattern for summer)
    hours = np.arange(horizon)
    base_price = 50 + 30 * np.sin((hours - 6) * np.pi / 12)  # Peak around 6 PM
    spot_prices = base_price + np.random.randn(horizon) * 5
    spot_prices = np.maximum(spot_prices, 10)  # Floor at $10

    optimization_results = energy_cost_optimization(
        samples,
        spot_prices,
        storage_capacity_mwh=1000.0,
        storage_efficiency=0.85,
        max_charge_rate=250.0,
    )
    print(f"   Net benefit: ${optimization_results['net_benefit']:,.0f}")

    # ==== Generate Visualizations ====
    print("\nGenerating visualizations...")
    plot_peak_detection(context.values, samples, peak_results, dr_results, output_dir)
    plot_reserve_analysis(reserve_results, output_dir)
    plot_storage_optimization(optimization_results, spot_prices, output_dir)

    # ==== Summary ====
    create_prescriptive_summary(
        peak_results, dr_results, reserve_results, optimization_results, output_dir
    )

    print("\n" + "=" * 70)
    print("PRESCRIPTIVE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
