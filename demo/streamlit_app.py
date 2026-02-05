"""Interactive Streamlit demo for the energy load forecasting benchmark.

Launch with:
    streamlit run demo/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Energy Load Forecasting Benchmark",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Load Forecasting Benchmark")
st.markdown(
    "Compare **Time Series Foundation Models** against statistical baselines "
    "on ERCOT hourly load data."
)

# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")

available_models = [
    "Seasonal Naive",
    "Chronos-Bolt (base)",
    "Chronos-2",
]

selected_models = st.sidebar.multiselect(
    "Models to compare",
    available_models,
    default=["Seasonal Naive"],
)

horizon = st.sidebar.selectbox(
    "Forecast horizon (hours)",
    [24, 168, 720],
    index=0,
)

context_length = st.sidebar.selectbox(
    "Context length (hours)",
    [512, 1024],
    index=0,
)

num_windows = st.sidebar.slider(
    "Number of rolling windows",
    min_value=1,
    max_value=50,
    value=10,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading ERCOT data...")
def load_data():
    from energy_benchmark.data import ERCOTLoader
    from energy_benchmark.data.preprocessing import preprocess_series

    loader = ERCOTLoader(years=[2020, 2021, 2022, 2023, 2024])
    series = loader.load()
    series = preprocess_series(series)
    train, val, test = loader.split(series)
    return series, train, val, test


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
tab_data, tab_benchmark, tab_results = st.tabs([
    "📊 Data Explorer",
    "🚀 Run Benchmark",
    "📈 Results",
])

# ---- Tab 1: Data Explorer ------------------------------------------------
with tab_data:
    try:
        series, train, val, test = load_data()

        col1, col2, col3 = st.columns(3)
        col1.metric("Train hours", f"{len(train):,}")
        col2.metric("Validation hours", f"{len(val):,}")
        col3.metric("Test hours", f"{len(test):,}")

        st.subheader("Full Time Series")
        fig, ax = plt.subplots(figsize=(14, 4))
        train.plot(ax=ax, label="Train", alpha=0.7, linewidth=0.4)
        val.plot(ax=ax, label="Validation", alpha=0.7, linewidth=0.4)
        test.plot(ax=ax, label="Test", alpha=0.7, linewidth=0.4)
        ax.set_ylabel("Load (MW)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Daily Load Profile")
        hourly_avg = series.groupby(series.index.hour).mean()
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(hourly_avg.index, hourly_avg.values, "o-")
        ax2.set_xlabel("Hour of day")
        ax2.set_ylabel("Mean load (MW)")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    except Exception as e:
        st.error(
            f"Could not load data: {e}\n\n"
            "Run `python scripts/download_data.py` to fetch ERCOT data from the "
            "EIA Open Data API.  If you hit rate limits with the default DEMO_KEY, "
            "set the `EIA_API_KEY` environment variable with your free key from "
            "https://www.eia.gov/opendata/register.php"
        )

# ---- Tab 2: Run Benchmark ------------------------------------------------
with tab_benchmark:
    if st.button("▶ Run Benchmark", type="primary"):
        if not selected_models:
            st.warning("Select at least one model from the sidebar.")
        else:
            try:
                series, train, val, test = load_data()
            except Exception as e:
                st.error(f"Data not available: {e}")
                st.stop()

            from energy_benchmark.models import SeasonalNaiveModel
            from energy_benchmark.evaluation import BenchmarkRunner

            models = []
            for name in selected_models:
                if name == "Seasonal Naive":
                    m = SeasonalNaiveModel(seasonality=168)
                    m.fit(train)
                    models.append(m)
                elif name == "Chronos-Bolt (base)":
                    try:
                        from energy_benchmark.models import ChronosBoltModel
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        m = ChronosBoltModel(model_size="base", device=device)
                        m.fit(train)
                        models.append(m)
                    except ImportError:
                        st.warning("chronos-forecasting not installed, skipping Chronos-Bolt")
                elif name == "Chronos-2":
                    try:
                        from energy_benchmark.models import Chronos2Model
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        m = Chronos2Model(device=device)
                        m.fit(train)
                        models.append(m)
                    except ImportError:
                        st.warning("chronos-forecasting not installed, skipping Chronos-2")

            if not models:
                st.error("No models could be initialised.")
                st.stop()

            runner = BenchmarkRunner(
                models=models,
                prediction_horizons=[horizon],
                context_lengths=[context_length],
                num_samples=50,
                metric_names=["mae", "rmse", "mase"],
            )

            with st.spinner("Running benchmark..."):
                results = runner.run(
                    train, test,
                    rolling_config={"step_size": 24, "num_windows": num_windows},
                )

            st.session_state["results"] = results
            st.success("Benchmark completed!")

# ---- Tab 3: Results ------------------------------------------------------
with tab_results:
    if "results" not in st.session_state:
        st.info("Run the benchmark first (tab 🚀) to see results here.")
    else:
        results = st.session_state["results"]
        df = results.to_dataframe()

        st.subheader("Results Table")
        st.dataframe(df, use_container_width=True)

        st.subheader("Model Comparison")
        for metric in ["mae", "rmse", "mase"]:
            if metric in df.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                df_plot = df[["model", metric]].set_index("model")
                df_plot.plot(kind="bar", ax=ax, legend=False)
                ax.set_ylabel(metric.upper())
                ax.set_title(f"{metric.upper()} by Model")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.subheader("Example Forecast")
        for fr in results.forecasts:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(fr.actual, "k-", linewidth=1.5, label="Actual")
            ax.plot(fr.point_forecast, "b-", linewidth=1, label="Forecast")
            if fr.samples is not None:
                lower = np.quantile(fr.samples, 0.1, axis=0)
                upper = np.quantile(fr.samples, 0.9, axis=0)
                ax.fill_between(
                    range(len(lower)), lower, upper,
                    alpha=0.2, color="blue", label="10–90% CI",
                )
            ax.set_title(f"{fr.model_name} — {fr.horizon}h (window {fr.window_idx})")
            ax.set_ylabel("Load (MW)")
            ax.set_xlabel("Hour")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            break  # show just the first window
