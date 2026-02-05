"""Script to download ERCOT data via the EIA Open Data API and cache it locally.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --years 2023 2024
    python scripts/download_data.py --data-dir data/raw --force
    python scripts/download_data.py --api-key YOUR_EIA_KEY
"""

import argparse
import logging

from energy_benchmark.data import ERCOTLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ERCOT load data via EIA API"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024],
        help="Years to download (default: 2020-2024)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory to store downloaded data",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="EIA API key (default: env var EIA_API_KEY or DEMO_KEY)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    args = parser.parse_args()

    loader = ERCOTLoader(
        years=args.years,
        data_dir=args.data_dir,
        api_key=args.api_key,
    )
    series = loader.load(force_download=args.force)

    print(f"\nLoaded {len(series)} hourly observations")
    print(f"Date range: {series.index.min()} — {series.index.max()}")
    print(f"\nBasic statistics:\n{series.describe()}")


if __name__ == "__main__":
    main()
