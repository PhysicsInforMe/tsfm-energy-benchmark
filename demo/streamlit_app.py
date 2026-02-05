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
    "Compare **Time Series Foundation Models** (Chronos-Bolt, Chronos-2, Lag-Llama) "
    "against **statistical baselines** (Seasonal Naive, ARIMA, Prophet) "
    "on ERCOT hourly load data."
)

# ---------------------------------------------------------------------------
# Model registry — detect what is installed
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {}


def _check_available():
    """Populate MODEL_REGISTRY with all models that can be instantiated."""
    # Always available (no extra deps)
    MODEL_REGISTRY["Seasonal Naive"] = {
        "type": "baseline",
        "available": True,
        "missing_pkg": None,
    }

    # ARIMA — needs pmdarima
    try:
        import pmdarima  # noqa: F401
        MODEL_REGISTRY["ARIMA"] = {
            "type": "baseline",
            "available": True,
            "missing_pkg": None,
        }
    except ImportError:
        MODEL_REGISTRY["ARIMA"] = {
            "type": "baseline",
            "available": False,
            "missing_pkg": "pmdarima",
        }

    # Prophet
    try:
        import prophet  # noqa: F401
        MODEL_REGISTRY["Prophet"] = {
            "type": "baseline",
            "available": True,
            "missing_pkg": None,
        }
    except ImportError:
        MODEL_REGISTRY["Prophet"] = {
            "type": "baseline",
            "available": False,
            "missing_pkg": "prophet",
        }

    # Chronos-Bolt — needs chronos-forecasting
    try:
        from chronos import ChronosBoltPipeline  # noqa: F401
        MODEL_REGISTRY["Chronos-Bolt"] = {
            "type": "foundation",
            "available": True,
            "missing_pkg": None,
        }
    except ImportError:
        MODEL_REGISTRY["Chronos-Bolt"] = {
            "type": "foundation",
            "available": False,
            "missing_pkg": "chronos-forecasting",
        }

    # Chronos-2 — needs chronos-forecasting
    try:
        from chronos import ChronosPipeline  # noqa: F401
        MODEL_REGISTRY["Chronos-2"] = {
            "type": "foundation",
            "available": True,
            "missing_pkg": None,
        }
    except ImportError:
        MODEL_REGISTRY["Chronos-2"] = {
            "type": "foundation",
            "available": False,
            "missing_pkg": "chronos-forecasting",
        }

    # Lag-Llama — needs lag-llama repo + gluonts
    try:
        import gluonts  # noqa: F401
        MODEL_REGISTRY["Lag-Llama"] = {
            "type": "foundation",
            "available": True,
            "missing_pkg": None,
        }
    except ImportError:
        MODEL_REGISTRY["Lag-Llama"] = {
            "type": "foundation",
            "available": False,
            "missing_pkg": "gluonts (+ lag-llama repo)",
        }


_check_available()

# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")

# Show available vs unavailable models
available_names = [k for k, v in MODEL_REGISTRY.items() if v["available"]]
unavailable = {k: v["missing_pkg"] for k, v in MODEL_REGISTRY.items() if not v["available"]}

# Foundation models first, then baselines
foundation = [n for n in available_names if MODEL_REGISTRY[n]["type"] == "foundation"]
baselines = [n for n in available_names if MODEL_REGISTRY[n]["type"] == "baseline"]
ordered = foundation + baselines

selected_models = st.sidebar.multiselect(
    "Models to compare",
    ordered,
    default=ordered,  # select all available by default
)

if unavailable:
    with st.sidebar.expander(f"{len(unavailable)} model(s) not installed"):
        for name, pkg in unavailable.items():
            mtype = MODEL_REGISTRY[name]["type"]
            st.caption(f"**{name}** ({mtype}) — `pip install {pkg}`")

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

    # Use contiguous years to ensure valid context windows
    data_dir = Path("data/raw")
    loader = ERCOTLoader(years=[2023, 2024], data_dir=str(data_dir))
    series = loader.load()
    series = preprocess_series(series)
    train, val, test = loader.split(
        series,
        train_end="2023-12-31",
        val_end="2024-03-31",
    )
    return series, train, val, test


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_models(names: list[str], train: pd.Series) -> list:
    """Instantiate and fit the selected models."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []

    for name in names:
        try:
            if name == "Seasonal Naive":
                from energy_benchmark.models import SeasonalNaiveModel
                m = SeasonalNaiveModel(seasonality=168)
                m.fit(train)
                models.append(m)

            elif name == "ARIMA":
                from energy_benchmark.models import ARIMAModel
                m = ARIMAModel(
                    order=(2, 1, 2),
                    seasonal_order=(1, 1, 1, 24),
                )
                m.fit(train)
                models.append(m)

            elif name == "Prophet":
                from energy_benchmark.models import ProphetModel
                m = ProphetModel(max_train_hours=8760)
                m.fit(train)
                models.append(m)

            elif name == "Chronos-Bolt":
                from energy_benchmark.models import ChronosBoltModel
                m = ChronosBoltModel(model_size="base", device=device)
                m.fit(train)
                models.append(m)

            elif name == "Chronos-2":
                from energy_benchmark.models import Chronos2Model
                m = Chronos2Model(device=device)
                m.fit(train)
                models.append(m)

            elif name == "Lag-Llama":
                from energy_benchmark.models import LagLlamaModel
                m = LagLlamaModel(context_length=512, device=device)
                m.fit(train)
                models.append(m)

        except Exception as e:
            st.warning(f"Could not initialise **{name}**: {e}")

    return models


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
tab_data, tab_benchmark, tab_results = st.tabs([
    "Data Explorer",
    "Run Benchmark",
    "Results",
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
    st.markdown(
        f"**Selected models**: {', '.join(selected_models) if selected_models else 'None'} "
        f"| **Horizon**: {horizon}h | **Context**: {context_length}h | "
        f"**Windows**: {num_windows}"
    )

    n_foundation = sum(
        1 for n in selected_models
        if MODEL_REGISTRY.get(n, {}).get("type") == "foundation"
    )
    n_baseline = sum(
        1 for n in selected_models
        if MODEL_REGISTRY.get(n, {}).get("type") == "baseline"
    )
    if n_foundation == 0 and selected_models:
        st.info(
            "No foundation models selected. Select at least one "
            "(Chronos-Bolt, Chronos-2, Lag-Llama) from the sidebar to "
            "compare against baselines."
        )

    if st.button("Run Benchmark", type="primary"):
        if not selected_models:
            st.warning("Select at least one model from the sidebar.")
        else:
            try:
                series, train, val, test = load_data()
            except Exception as e:
                st.error(f"Data not available: {e}")
                st.stop()

            from energy_benchmark.evaluation import BenchmarkRunner

            with st.spinner("Initialising models..."):
                models = build_models(selected_models, train)

            if not models:
                st.error("No models could be initialised.")
                st.stop()

            st.info(
                f"Running benchmark with **{len(models)} models**: "
                f"{[m.name for m in models]}"
            )

            runner = BenchmarkRunner(
                models=models,
                prediction_horizons=[horizon],
                context_lengths=[context_length],
                num_samples=50,
                metric_names=["mae", "rmse", "mase"],
            )

            progress = st.progress(0, text="Running rolling evaluation...")
            with st.spinner("Running benchmark..."):
                results = runner.run(
                    train, test,
                    rolling_config={"step_size": 24, "num_windows": num_windows},
                )
            progress.progress(100, text="Done!")

            st.session_state["results"] = results
            st.success(
                f"Benchmark completed! {len(results.records)} result rows."
            )

# ---- Tab 3: Results ------------------------------------------------------
with tab_results:
    if "results" not in st.session_state:
        st.info("Run the benchmark first (tab Run Benchmark) to see results here.")
    else:
        results = st.session_state["results"]
        df = results.to_dataframe()

        st.subheader("Results Table")
        st.dataframe(df, use_container_width=True)

        # ---- Summary metrics by model ----
        st.subheader("Model Comparison")
        metric_cols = [c for c in ["mae", "rmse", "mase"] if c in df.columns]

        if metric_cols:
            summary = df.groupby("model")[metric_cols].mean().sort_values(
                metric_cols[0]
            )
            st.dataframe(
                summary.style.format("{:.2f}").background_gradient(
                    cmap="YlOrRd", axis=0
                ),
                use_container_width=True,
            )

        # ---- Bar charts per metric ----
        for metric in metric_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            model_means = df.groupby("model")[metric].mean().sort_values()
            colors = [
                "#2196F3" if MODEL_REGISTRY.get(m, {}).get("type") == "foundation"
                else "#FF9800"
                for m in model_means.index
            ]
            model_means.plot(kind="barh", ax=ax, color=colors)
            ax.set_xlabel(metric.upper())
            ax.set_title(f"{metric.upper()} by Model (lower is better)")

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#2196F3", label="Foundation Model"),
                Patch(facecolor="#FF9800", label="Statistical Baseline"),
            ]
            ax.legend(handles=legend_elements, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ---- MASE interpretation ----
        if "mase" in df.columns:
            st.subheader("MASE Interpretation")
            mase_by_model = df.groupby("model")["mase"].mean().sort_values()
            for model_name, mase_val in mase_by_model.items():
                if mase_val < 1.0:
                    pct = (1.0 - mase_val) * 100
                    st.success(
                        f"**{model_name}**: MASE = {mase_val:.3f} "
                        f"-- {pct:.1f}% better than Seasonal Naive"
                    )
                elif mase_val == 1.0:
                    st.warning(
                        f"**{model_name}**: MASE = {mase_val:.3f} "
                        f"-- on par with Seasonal Naive"
                    )
                else:
                    pct = (mase_val - 1.0) * 100
                    st.error(
                        f"**{model_name}**: MASE = {mase_val:.3f} "
                        f"-- {pct:.1f}% worse than Seasonal Naive"
                    )

        # ---- Example forecasts ----
        st.subheader("Example Forecasts")
        # Show one forecast per model
        shown_models = set()
        for fr in results.forecasts:
            if fr.model_name in shown_models:
                continue
            shown_models.add(fr.model_name)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(fr.actual, "k-", linewidth=1.5, label="Actual")
            ax.plot(fr.point_forecast, "b-", linewidth=1, label="Forecast")
            if fr.samples is not None:
                lower = np.quantile(fr.samples, 0.1, axis=0)
                upper = np.quantile(fr.samples, 0.9, axis=0)
                ax.fill_between(
                    range(len(lower)), lower, upper,
                    alpha=0.2, color="blue", label="10-90% CI",
                )
            mtype = MODEL_REGISTRY.get(fr.model_name, {}).get("type", "")
            tag = "[Foundation]" if mtype == "foundation" else "[Baseline]"
            ax.set_title(
                f"{fr.model_name} {tag} -- {fr.horizon}h forecast"
            )
            ax.set_ylabel("Load (MW)")
            ax.set_xlabel("Hour")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
