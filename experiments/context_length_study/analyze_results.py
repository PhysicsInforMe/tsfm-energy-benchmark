"""
Statistical Analysis and Visualization for Context Length Experiment

Produces publication-ready figures and statistical tests.
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
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color scheme
COLORS = {
    "Chronos-Bolt": "#2196F3",  # Blue
    "Chronos-2": "#4CAF50",      # Green
    "SARIMA": "#FF9800",         # Orange
    "SeasonalNaive": "#9E9E9E",  # Gray
}

MODEL_ORDER = ["Chronos-Bolt", "Chronos-2", "SARIMA", "SeasonalNaive"]


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load combined results."""
    df = pd.read_csv(results_dir / "results_all_models.csv")
    return df


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, 95% CI for each condition."""
    summary = df.groupby(["model", "context_length", "horizon", "period"]).agg({
        "mase": ["mean", "std", "count"],
        "mae": ["mean", "std"],
        "rmse": ["mean", "std"],
        "inference_seconds": ["mean", "std"],
    }).round(4)

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Compute 95% CI
    summary["mase_ci95"] = 1.96 * summary["mase_std"] / np.sqrt(summary["mase_count"])

    return summary


def plot_mase_vs_context(df: pd.DataFrame, output_dir: Path):
    """Figure 1: MASE vs Context Length curves."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, horizon in zip(axes, [24, 168]):
        subset = df[df["horizon"] == horizon]

        for model in MODEL_ORDER:
            model_data = subset[subset["model"] == model]
            agg = model_data.groupby("context_length")["mase"].agg(["mean", "std", "count"])
            agg["ci95"] = 1.96 * agg["std"] / np.sqrt(agg["count"])

            ax.errorbar(
                agg.index, agg["mean"], yerr=agg["ci95"],
                marker="o", markersize=5, linewidth=1.5,
                color=COLORS[model], label=model,
                capsize=3, capthick=1,
            )

        ax.set_xlabel("Context Length (hours)")
        ax.set_ylabel("MASE")
        ax.set_title(f"Horizon = {horizon}h")
        ax.set_xscale("log", base=2)
        ax.set_xticks([24, 48, 96, 168, 336, 512, 1024, 2048])
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, label="Naive baseline")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    # Adjust y-limits to focus on reasonable range
    axes[0].set_ylim(0, 2.5)
    axes[1].set_ylim(0, 4)  # SARIMA has higher errors for h=168

    fig.suptitle("Context Length Sensitivity: MASE vs Context Length", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_mase_vs_context.png")
    fig.savefig(output_dir / "fig1_mase_vs_context.pdf")
    plt.close()
    print(f"Saved: fig1_mase_vs_context.png/pdf")


def plot_mase_heatmap(df: pd.DataFrame, output_dir: Path):
    """Figure 2: Heatmap of MASE by model, context, period."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    periods = ["summer_2023", "winter_2022", "covid_2020"]
    period_titles = ["Summer 2023", "Winter 2022-23", "COVID March 2020"]

    for ax, period, title in zip(axes, periods, period_titles):
        subset = df[(df["period"] == period) & (df["horizon"] == 24)]
        pivot = subset.pivot_table(
            values="mase", index="model", columns="context_length", aggfunc="mean"
        )
        pivot = pivot.reindex(MODEL_ORDER)

        # Clip for visualization (SARIMA has extreme values at short contexts)
        pivot_clipped = pivot.clip(upper=3.0)

        im = ax.imshow(pivot_clipped, cmap="RdYlGn_r", aspect="auto", vmin=0.3, vmax=3.0)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Context Length (hours)")
        ax.set_title(title)

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                text = f"{val:.2f}" if val < 10 else f"{val:.1f}"
                color = "white" if val > 1.5 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    axes[0].set_ylabel("Model")
    fig.colorbar(im, ax=axes, shrink=0.8, label="MASE")
    fig.suptitle("MASE Heatmap by Model, Context Length, and Test Period (h=24)", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fig2_mase_heatmap.png")
    fig.savefig(output_dir / "fig2_mase_heatmap.pdf")
    plt.close()
    print(f"Saved: fig2_mase_heatmap.png/pdf")


def plot_inference_time(df: pd.DataFrame, output_dir: Path):
    """Figure 3: Inference time vs Context Length."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for model in MODEL_ORDER:
        if model == "SeasonalNaive":
            continue  # Skip - effectively zero
        model_data = df[df["model"] == model]
        agg = model_data.groupby("context_length")["inference_seconds"].mean()

        ax.plot(
            agg.index, agg.values,
            marker="o", markersize=5, linewidth=1.5,
            color=COLORS[model], label=model,
        )

    ax.set_xlabel("Context Length (hours)")
    ax.set_ylabel("Inference Time (seconds)")
    ax.set_title("Computational Cost: Inference Time vs Context Length")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([24, 48, 96, 168, 336, 512, 1024, 2048])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_inference_time.png")
    fig.savefig(output_dir / "fig3_inference_time.pdf")
    plt.close()
    print(f"Saved: fig3_inference_time.png/pdf")


def plot_model_comparison_bar(df: pd.DataFrame, output_dir: Path):
    """Figure 4: Bar chart comparing models at best context."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Use context=512 for fair comparison (all models have results)
    subset = df[(df["context_length"] == 512) & (df["horizon"] == 24)]

    means = subset.groupby("model")["mase"].mean()
    stds = subset.groupby("model")["mase"].std()
    counts = subset.groupby("model")["mase"].count()
    ci95 = 1.96 * stds / np.sqrt(counts)

    means = means.reindex(MODEL_ORDER)
    ci95 = ci95.reindex(MODEL_ORDER)

    x = np.arange(len(MODEL_ORDER))
    colors = [COLORS[m] for m in MODEL_ORDER]

    bars = ax.bar(x, means, yerr=ci95, color=colors, capsize=5, alpha=0.8)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Naive baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15)
    ax.set_ylabel("MASE")
    ax.set_title("Model Comparison at Context=512h, Horizon=24h")
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_model_comparison.png")
    fig.savefig(output_dir / "fig4_model_comparison.pdf")
    plt.close()
    print(f"Saved: fig4_model_comparison.png/pdf")


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Table 1: Summary statistics."""
    # Overall summary by model
    summary = df.groupby("model").agg({
        "mase": ["mean", "std", "min"],
        "mae": ["mean"],
        "rmse": ["mean"],
        "inference_seconds": ["mean"],
    }).round(3)

    summary.columns = ["MASE Mean", "MASE Std", "MASE Min", "MAE Mean", "RMSE Mean", "Inference (s)"]
    summary = summary.reindex(MODEL_ORDER)

    print("\n" + "="*70)
    print("TABLE 1: Summary Statistics by Model (All Conditions)")
    print("="*70)
    print(summary.to_string())

    summary.to_csv(output_dir / "table1_summary.csv")
    print(f"\nSaved: table1_summary.csv")

    # Best context per model
    print("\n" + "="*70)
    print("TABLE 2: Best Context Length per Model")
    print("="*70)

    best_ctx = []
    for model in MODEL_ORDER:
        model_data = df[df["model"] == model]
        ctx_means = model_data.groupby("context_length")["mase"].mean()
        best = ctx_means.idxmin()
        best_mase = ctx_means.min()
        worst = ctx_means.idxmax()
        worst_mase = ctx_means.max()
        improvement = (worst_mase - best_mase) / worst_mase * 100
        best_ctx.append({
            "Model": model,
            "Best Context": best,
            "Best MASE": round(best_mase, 3),
            "Worst Context": worst,
            "Worst MASE": round(worst_mase, 3),
            "Improvement %": round(improvement, 1),
        })

    best_ctx_df = pd.DataFrame(best_ctx)
    print(best_ctx_df.to_string(index=False))
    best_ctx_df.to_csv(output_dir / "table2_best_context.csv", index=False)
    print(f"\nSaved: table2_best_context.csv")


def statistical_tests(df: pd.DataFrame, output_dir: Path):
    """Perform statistical tests."""
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    # Test: Do foundation models beat baselines at optimal context?
    ctx_512 = df[(df["context_length"] == 512) & (df["horizon"] == 24)]

    chronos_bolt = ctx_512[ctx_512["model"] == "Chronos-Bolt"]["mase"]
    chronos_2 = ctx_512[ctx_512["model"] == "Chronos-2"]["mase"]
    seasonal = ctx_512[ctx_512["model"] == "SeasonalNaive"]["mase"]
    sarima = ctx_512[ctx_512["model"] == "SARIMA"]["mase"]

    print("\n1. T-test: Chronos-Bolt vs SeasonalNaive (ctx=512, h=24)")
    t_stat, p_val = stats.ttest_ind(chronos_bolt, seasonal)
    print(f"   t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
    print(f"   Significant at alpha=0.05: {'Yes' if p_val < 0.05 else 'No'}")

    print("\n2. T-test: Chronos-2 vs SeasonalNaive (ctx=512, h=24)")
    t_stat, p_val = stats.ttest_ind(chronos_2, seasonal)
    print(f"   t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
    print(f"   Significant at alpha=0.05: {'Yes' if p_val < 0.05 else 'No'}")

    print("\n3. T-test: Chronos-Bolt vs SARIMA (ctx=512, h=24)")
    t_stat, p_val = stats.ttest_ind(chronos_bolt, sarima)
    print(f"   t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
    print(f"   Significant at alpha=0.05: {'Yes' if p_val < 0.05 else 'No'}")

    # Effect of context length on foundation models
    print("\n4. Spearman correlation: Context Length vs MASE (Chronos-Bolt)")
    cb_data = df[df["model"] == "Chronos-Bolt"]
    corr, p_val = stats.spearmanr(cb_data["context_length"], cb_data["mase"])
    print(f"   Correlation: {corr:.3f}, p-value: {p_val:.4f}")
    print(f"   Interpretation: {'Longer context reduces MASE' if corr < 0 else 'No clear trend'}")


def main():
    """Run complete analysis."""
    results_dir = Path(__file__).parent / "results" / "raw"
    output_dir = Path(__file__).parent / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_results(results_dir)
    print(f"Loaded {len(df)} results for {df['model'].nunique()} models")

    # Generate figures
    print("\nGenerating figures...")
    plot_mase_vs_context(df, output_dir)
    plot_mase_heatmap(df, output_dir)
    plot_inference_time(df, output_dir)
    plot_model_comparison_bar(df, output_dir)

    # Create tables
    create_summary_table(df, output_dir)

    # Statistical tests
    statistical_tests(df, output_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
