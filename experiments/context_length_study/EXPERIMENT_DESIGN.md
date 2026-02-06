# Context Length Sensitivity in Time Series Foundation Models
## Experimental Design Document

### Research Question

**Primary RQ:** How does context length affect forecast accuracy in time series foundation models compared to statistical baselines?

**Secondary RQs:**
1. Is there a saturation point beyond which more context doesn't help (or hurts)?
2. Do foundation models and statistical models respond differently to context length?
3. Does the optimal context length vary by forecast horizon or seasonal period?

### Hypotheses

**H1:** Foundation models will show diminishing returns beyond a certain context length (saturation hypothesis)

**H2:** Statistical models (ARIMA) will degrade with very long contexts due to parameter estimation noise

**H3:** Optimal context length will be shorter for short-term forecasts (24h) than for long-term (168h)

### Experimental Design

#### Independent Variables

1. **Context Length** (8 levels): 24, 48, 96, 168, 336, 512, 1024, 2048 hours
   - 24h = 1 day (minimal)
   - 168h = 1 week (captures weekly seasonality)
   - 2048h = 85 days (maximum practical)

2. **Forecast Horizon** (2 levels): 24h (day-ahead), 168h (week-ahead)
   - Day-ahead: standard market requirement
   - Week-ahead: planning horizon

3. **Test Period** (3 levels):
   - Summer 2023 (high demand, cooling load)
   - Winter 2022-23 (heating load, includes winter storm)
   - COVID period March 2020 (demand shock, distribution shift)

#### Models (4 total)

1. **Chronos-Bolt-Small** — Fast foundation model, encoder-decoder
2. **Chronos-2** — Latest foundation model, encoder-only with group attention
3. **Seasonal Naive (168h)** — Baseline: repeat last week
4. **SARIMA(2,1,2)(1,1,1,24)** — Classical statistical with daily seasonality

#### Dependent Variables

1. **MASE** — Primary metric (scale-free, interpretable vs naive)
2. **MAE** — Absolute error in MW
3. **RMSE** — Penalizes large errors
4. **Inference Time** — Seconds per forecast (computational cost)

#### Experimental Protocol

1. **Rolling Window Evaluation**
   - For each (context_length, horizon, test_period) combination
   - Use 10 rolling windows, step size = 24h
   - Report mean and std of metrics across windows

2. **Data Splits**
   - Training data: 2020-01-01 to 2022-12-31 (for MASE scaling factor)
   - Context: drawn from data preceding each test window
   - Test periods defined above

3. **Statistical Analysis**
   - Mean ± 95% CI for each condition
   - Two-way ANOVA: context_length × model interaction
   - Post-hoc pairwise comparisons with Bonferroni correction

### Sample Size Calculation

- 8 context lengths × 2 horizons × 3 periods × 4 models = 192 conditions
- 10 windows per condition = 1,920 total forecasts
- Sufficient for detecting medium effect sizes (Cohen's d > 0.5)

### Expected Outputs

1. **Figure 1:** MASE vs Context Length curves (one line per model)
2. **Figure 2:** Heatmap of optimal context length by (model, horizon, period)
3. **Figure 3:** Inference time vs Context Length
4. **Table 1:** Summary statistics for all conditions
5. **Table 2:** ANOVA results and significant interactions

### Limitations (to acknowledge in paper)

- CPU-only inference (may affect absolute timings)
- Single dataset (ERCOT Texas)
- Limited to publicly available models
- No fine-tuning conditions (zero-shot only)

### File Structure

```
experiments/context_length_study/
    EXPERIMENT_DESIGN.md      # This document
    run_experiment.py         # Main experiment script
    analyze_results.py        # Statistical analysis
    results/
        raw/                  # Raw forecast outputs
        processed/            # Aggregated metrics
        figures/              # Publication figures
    paper/
        main.tex              # Paper draft
        figures/              # Final figures
```
