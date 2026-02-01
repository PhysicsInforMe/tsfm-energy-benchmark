"""Run the full energy forecasting benchmark.

Usage:
    python scripts/run_benchmark.py --config configs/benchmark_config.yaml
    python scripts/run_benchmark.py --config configs/benchmark_config.yaml --cpu
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
import pandas as pd

from energy_benchmark.data import ERCOTLoader
from energy_benchmark.data.preprocessing import preprocess_series
from energy_benchmark.models import (
    SeasonalNaiveModel,
    ARIMAModel,
    ChronosBoltModel,
    Chronos2Model,
    LagLlamaModel,
)
from energy_benchmark.evaluation import BenchmarkRunner
from energy_benchmark.visualization import (
    plot_comparison,
    plot_metric_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_models(model_cfg: dict, force_cpu: bool = False) -> list:
    """Instantiate enabled models from config."""
    models = []
    device = "cpu" if force_cpu else None

    if model_cfg.get("seasonal_naive", {}).get("enabled", False):
        models.append(
            SeasonalNaiveModel(
                seasonality=model_cfg["seasonal_naive"].get("seasonality", 168)
            )
        )

    if model_cfg.get("arima", {}).get("enabled", False):
        models.append(
            ARIMAModel(
                order=tuple(model_cfg["arima"].get("order", [2, 1, 2])),
                seasonal_order=tuple(
                    model_cfg["arima"].get("seasonal_order", [1, 1, 1, 24])
                ),
            )
        )

    if model_cfg.get("chronos_bolt", {}).get("enabled", False):
        models.append(
            ChronosBoltModel(
                model_size=model_cfg["chronos_bolt"].get("size", "base"),
                device=device or model_cfg["chronos_bolt"].get("device", "cuda"),
            )
        )

    if model_cfg.get("chronos_2", {}).get("enabled", False):
        models.append(
            Chronos2Model(
                device=device or model_cfg["chronos_2"].get("device", "cuda"),
            )
        )

    if model_cfg.get("lag_llama", {}).get("enabled", False):
        models.append(
            LagLlamaModel(
                context_length=model_cfg["lag_llama"].get("context_length", 512),
                device=device or "cuda",
            )
        )

    return models


def main(config_path: str, force_cpu: bool = False) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", config_path)

    # 1. Load and preprocess data
    data_cfg = config["data"]
    loader = ERCOTLoader(
        years=data_cfg["years"],
        data_dir=data_cfg.get("data_dir", "data/raw"),
    )
    series = loader.load()
    series = preprocess_series(series)

    train, val, test = loader.split(
        series,
        train_end=data_cfg["split"]["train_end"],
        val_end=data_cfg["split"]["val_end"],
    )
    logger.info(
        "Data split — train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )

    # 2. Build models
    models = _build_models(config["models"], force_cpu=force_cpu)
    if not models:
        logger.error("No models enabled in config. Exiting.")
        return
    logger.info("Models: %s", [m.name for m in models])

    # 3. Run benchmark
    bench_cfg = config["benchmark"]
    runner = BenchmarkRunner(
        models=models,
        prediction_horizons=bench_cfg["prediction_horizons"],
        context_lengths=bench_cfg["context_lengths"],
        num_samples=bench_cfg.get("num_samples", 100),
        metric_names=config.get("metrics", ["mae", "rmse", "mase"]),
    )

    results = runner.run(train, test, rolling_config=bench_cfg.get("rolling"))

    # 4. Save results
    output_cfg = config["output"]
    output_dir = Path(output_cfg["results_dir"])
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = results.to_dataframe()
    csv_path = tables_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    # 5. Generate figures
    if output_cfg.get("save_figures", True):
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        for metric in config.get("metrics", ["mae"]):
            if metric in df.columns:
                plot_comparison(
                    df, metric=metric, save_path=fig_dir / f"comparison_{metric}.png"
                )
                plot_metric_heatmap(
                    df, metric=metric, save_path=fig_dir / f"heatmap_{metric}.png"
                )

        logger.info("Figures saved to %s", fig_dir)

    # 6. Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run energy forecasting benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmark_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force all models to run on CPU",
    )
    args = parser.parse_args()
    main(args.config, force_cpu=args.cpu)
