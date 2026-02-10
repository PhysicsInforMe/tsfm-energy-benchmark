# Context Length Sensitivity in Time Series Foundation Models

## Key Findings

This experiment investigates how historical context length affects forecast accuracy in time series foundation models compared to statistical baselines on ERCOT electricity load data.

### Main Results

| Model | Best Context | Best MASE | Parameters |
|-------|-------------|-----------|------------|
| Moirai-2 | 2048h | 0.307 | ~11M |
| Chronos-Bolt | 2048h | 0.315 | ~48M |
| Chronos-2 | 2048h | 0.334 | ~120M |
| SARIMA | 2048h | 0.370 | - |
| TTM | 2048h | 0.450 | <1M |
| Prophet | 2048h | 0.610 | - |
| Seasonal Naive | 24h | 0.749 | - |

### Key Insights

1. **Foundation models benefit from longer context**: All TSFMs show consistent improvement as context length increases from 24h to 2048h, with no saturation observed. Moirai-2 shows the steepest improvement (76% reduction in MASE).

2. **Pre-trained models have a structural advantage**: Prophet fails catastrophically at short context (MASE >74 at 24h) because it must estimate all parameters from scratch. TSFMs recognize patterns from pre-training and maintain stable accuracy even with minimal context.

3. **Foundation models are ~60x faster than SARIMA**: At context=512h, Chronos models take ~0.1s per forecast vs ~6.2s for SARIMA, making them practical for real-time applications.

4. **Uncertainty calibration varies**: Chronos-2 is well-calibrated (95% coverage at 90% nominal), while Moirai-2 and Prophet are overconfident (~70% coverage).

5. **Statistical significance**: Foundation models significantly outperform Seasonal Naive (p<0.001). The top three TSFMs (Chronos-Bolt, Chronos-2, Moirai-2) show statistically indistinguishable point accuracy (p>0.05).

## Experimental Design

- **Context Lengths**: 24, 48, 96, 168, 336, 512, 1024, 2048 hours
- **Horizons**: 24h (day-ahead), 168h (week-ahead)
- **Test Periods**: Summer 2023, Winter 2022-23, COVID March 2020
- **Models**: Chronos-Bolt (small), Chronos-2, Moirai-2 (small), TTM, Prophet, SARIMA(2,1,2)(1,1,1,24), Seasonal Naive (168h)
- **Evaluation**: 7 rolling windows per condition, 2,352 total forecast evaluations, MASE metric

## Files

```
experiments/context_length_study/
├── EXPERIMENT_DESIGN.md     # Formal experimental protocol
├── run_experiment.py        # Main experiment script
├── analyze_results.py       # Statistical analysis & visualization
├── results/
│   ├── raw/                 # Raw CSV results (1,680 observations)
│   └── figures/             # Publication-ready figures (PNG & PDF)
```

## Reproducing Results

```bash
# Run full experiment (~50 minutes on CPU)
python experiments/context_length_study/run_experiment.py

# Generate figures and tables
python experiments/context_length_study/analyze_results.py
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{tsfm-energy-benchmark,
  title={Time Series Foundation Models for Energy Load Forecasting
         on Consumer Hardware: A Multi-Dimensional Zero-Shot Benchmark},
  author={Luigi Simeone},
  year={2026},
  url={https://github.com/PhysicsInforMe/tsfm-energy-benchmark}
}
```
