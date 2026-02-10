# TSFM Energy Benchmark

A multi-dimensional benchmark evaluating **Time Series Foundation Models** for electricity demand forecasting on consumer-grade hardware. This project compares four TSFMs (Chronos-Bolt, Chronos-2, Moirai-2, TTM) against Prophet and statistical baselines using ERCOT data, across accuracy, calibration, robustness, and prescriptive utility dimensions.

## Key Findings

### Model Comparison

| Model | Best MASE | Inference Time | Type | Architecture |
|-------|-----------|----------------|------|--------------|
| **Chronos-Bolt** | 0.315 | 0.1s | Foundation Model | Encoder-Decoder |
| **Chronos-2** | 0.334 | 0.09s | Foundation Model | Encoder-only |
| **Moirai-2** | 0.307 | 0.025s | Foundation Model | Decoder-only |
| **TTM** | 0.450 | 0.003s | Foundation Model | MLP-Mixer |
| Prophet | 0.610 | 0.7s | Industry Baseline | Additive Decomposition |
| SARIMA | 0.370 | 6.2s | Statistical | Auto-regressive |
| Seasonal Naive | 0.749 | <0.01s | Baseline | Repetition |

### Main Insights

- **Foundation models achieve 47% lower error** than Seasonal Naive baseline at C=2048h
- **Pre-trained models have a structural advantage**: Prophet fails at short context (MASE >74 at C=24h) while TSFMs maintain stable accuracy
- **Longer context improves accuracy** for all models (no saturation up to 2048h)
- **Foundation models are ~60x faster** than SARIMA on CPU
- **Uncertainty calibration varies**: Chronos-2 is well-calibrated (95% coverage), Moirai-2 and Prophet are overconfident (~70% coverage)
- **Robustness varies under distribution shift**: Winter Storm Uri and holiday periods reveal model weaknesses

## Models Evaluated

### Foundation Models (Zero-Shot)

| Model | Paper | Source | Parameters |
|-------|-------|--------|------------|
| **Chronos-Bolt** (small) | [Ansari et al. 2024](https://arxiv.org/abs/2403.07815) | Amazon | ~48M |
| **Chronos-2** | [Ansari et al. 2025](https://arxiv.org/abs/2510.15821) | Amazon | ~120M |
| **Moirai-2** (small) | [Liu et al. 2025](https://arxiv.org/abs/2511.11698) | Salesforce | ~11M |
| **TinyTimeMixer (TTM)** | [IBM 2024](https://arxiv.org/abs/2401.03955) | IBM | <1M |

### Industry Baseline

| Model | Paper | Source |
|-------|-------|--------|
| **Prophet** | [Taylor & Letham 2018](https://peerj.com/preprints/3190/) | Meta |

### Statistical Baselines

- **SARIMA** - Seasonal ARIMA(2,1,2)(1,1,1,24)
- **Seasonal Naive** - Repeat last week (168h seasonality)

## Analyses Included

### 1. Context Length Sensitivity
How does forecast accuracy change with historical context?

### 2. Uncertainty Calibration
Are probabilistic forecasts well-calibrated? Do 90% prediction intervals contain 90% of observations?

### 3. Robustness Analysis
- Distribution shift (COVID, extreme weather)
- Temporal patterns (weekday vs weekend, seasonal)
- Missing data tolerance
- Extreme event performance

### 4. Prescriptive Analysis
- Peak load detection and demand response recommendations
- Optimal reserve margin planning
- Energy storage optimization under uncertainty

## Dataset

**ERCOT Hourly Load Data** (2020-2024)
- Source: EIA Open Data API
- 43,732 hourly observations
- Texas electricity grid demand

## Installation

```bash
# Clone repository
git clone https://github.com/PhysicsInforMe/tsfm-energy-benchmark.git
cd tsfm-energy-benchmark

# Install base dependencies
pip install -e ".[dev]"

# Install foundation model dependencies
pip install chronos-forecasting
pip install uni2ts

# For TinyTimeMixer (optional)
git clone https://github.com/ibm-granite/granite-tsfm.git
pip install ./granite-tsfm

# Download data
python scripts/download_data.py
```

## Quick Start

### Run All Experiments

```bash
# Context length sensitivity (main experiment)
python experiments/context_length_study/run_experiment.py

# Uncertainty calibration
python experiments/context_length_study/uncertainty_calibration.py

# Robustness analysis
python experiments/context_length_study/robustness_analysis.py

# Prescriptive analysis
python experiments/context_length_study/prescriptive_analysis.py

# Generate publication figures
python experiments/context_length_study/analyze_results.py
```

### Quick Test

```bash
# Verify setup (3 minutes)
python experiments/context_length_study/run_experiment.py --test
```

### Interactive Demo

```bash
streamlit run demo/streamlit_app.py
```

## Project Structure

```
tsfm-energy-benchmark/
├── src/energy_benchmark/
│   ├── data/                    # Data loaders (ERCOT via EIA API)
│   ├── models/                  # Model wrappers
│   │   ├── chronos_bolt.py      # Chronos-Bolt wrapper
│   │   ├── chronos2.py          # Chronos-2 wrapper
│   │   ├── moirai.py            # Moirai-2 wrapper
│   │   ├── tinytimemixer.py     # TTM wrapper
│   │   └── statistical.py       # SARIMA, Seasonal Naive
│   └── evaluation/              # Metrics and benchmark runner
├── experiments/
│   └── context_length_study/
│       ├── run_experiment.py           # Main experiment
│       ├── analyze_results.py          # Statistical analysis
│       ├── uncertainty_calibration.py  # Calibration analysis
│       ├── robustness_analysis.py      # Robustness tests
│       ├── prescriptive_analysis.py    # Prescriptive analytics
│       └── results/                    # Figures and data
├── demo/
│   └── streamlit_app.py                # Interactive demo
└── scripts/
    └── download_data.py                # Data download script
```

## Generated Outputs

### Figures

| Figure | Description |
|--------|-------------|
| fig1_mase_vs_context.png | MASE vs context length for all models |
| fig2_mase_heatmap.png | Performance heatmap by period |
| fig3_inference_time.png | Computational cost analysis |
| fig4_model_comparison.png | Model comparison at context=512 |
| fig5_calibration.png | Reliability diagram |
| fig6_prediction_intervals.png | Probabilistic forecast examples |
| fig7_peak_detection.png | Peak load detection |
| fig8_reserve_analysis.png | Reserve margin planning |
| fig9_storage_optimization.png | Energy storage optimization |
| fig10_distribution_shift.png | Performance under distribution shift |
| fig11_temporal_patterns.png | Weekday vs weekend performance |
| fig12_missing_data.png | Missing data robustness |
| fig13_extreme_events.png | Extreme event performance |

### Tables

| Table | Description |
|-------|-------------|
| table1_main_results.csv | Summary statistics for all conditions |
| table2_statistical_tests.csv | Hypothesis test results |
| table3_calibration.csv | Uncertainty calibration metrics |
| table4_prescriptive_summary.csv | Prescriptive analysis results |

## Experiment Details

See [experiments/context_length_study/README.md](experiments/context_length_study/README.md) for:
- Full experimental design
- Reproducibility instructions
- Statistical analysis methodology

## Hardware Requirements

- **Minimum**: 16GB RAM, CPU-only
- **Recommended**: 32GB RAM, NVIDIA GPU (CUDA)
- **Estimated runtime**: ~2 hours on CPU, ~30 minutes on GPU

## Citation

```bibtex
@misc{tsfm-energy-benchmark,
  title={Time Series Foundation Models for Energy Load Forecasting
         on Consumer Hardware: A Multi-Dimensional Zero-Shot Benchmark},
  author={Luigi Simeone},
  year={2026},
  url={https://github.com/PhysicsInforMe/tsfm-energy-benchmark}
}
```

## References

- Ansari et al. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815
- Ansari et al. (2025). Chronos-2: From Univariate to Universal Forecasting. arXiv:2510.15821
- Liu et al. (2025). Moirai 2.0: When Less Is More for Time Series Forecasting. arXiv:2511.11698
- Ekambaram et al. (2024). Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting. NeurIPS 2024
- Taylor & Letham (2018). Forecasting at Scale. The American Statistician 72(1):37-45

## License

MIT License

## Acknowledgments

- ERCOT data provided by EIA Open Data API
- Foundation models from Amazon, Salesforce, and IBM Research
- Prophet from Meta
