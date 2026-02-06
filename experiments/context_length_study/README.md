# Context Length Sensitivity in Time Series Foundation Models

## Key Findings

This experiment investigates how historical context length affects forecast accuracy in time series foundation models compared to statistical baselines on ERCOT electricity load data.

### Main Results

| Model | Best Context | Best MASE | Improvement |
|-------|-------------|-----------|-------------|
| Chronos-Bolt | 2048h | 0.315 | 42.6% vs worst |
| Chronos-2 | 2048h | 0.334 | 49.4% vs worst |
| SARIMA | 2048h | 0.365 | 97.6% vs worst |
| SeasonalNaive | 24h | 0.591 | 21.1% vs worst |

### Key Insights

1. **Foundation models benefit from longer context**: Both Chronos-Bolt and Chronos-2 show consistent improvement as context length increases from 24h to 2048h, with no saturation observed.

2. **SARIMA requires sufficient history**: SARIMA performs poorly with short contexts (MASE=15 at 24h) but becomes competitive with adequate data (MASE=0.37 at 2048h). This reflects the need for sufficient observations to estimate parameters.

3. **Foundation models are ~60x faster than SARIMA**: At context=512h, Chronos models take ~0.1s per forecast vs ~10s for SARIMA, making them practical for real-time applications.

4. **Statistical significance**: Foundation models significantly outperform SeasonalNaive (p<0.001) and SARIMA (p<0.01) at matched context lengths.

## Experimental Design

- **Context Lengths**: 24, 48, 96, 168, 336, 512, 1024, 2048 hours
- **Horizons**: 24h (day-ahead), 168h (week-ahead)
- **Test Periods**: Summer 2023, Winter 2022-23, COVID March 2020
- **Models**: Chronos-Bolt-Small, Chronos-2, SARIMA(2,1,2)(1,1,1,24), Seasonal Naive (168h)
- **Evaluation**: 10 rolling windows per condition, MASE metric

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
@misc{energy-load-forecasting-benchmark,
  title={Context Length Sensitivity in Time Series Foundation Models: An Energy Load Forecasting Benchmark},
  author={[Your Name]},
  year={2026},
  url={https://github.com/[your-repo]}
}
```
