from __future__ import annotations

import pandas as pd
import streamlit as st

from .analytics_db import query_df
from .asset_universe import read_asset_universe
from .utils import DUCKDB_PATH


DISCLAIMER = "Educational analytics only. This dashboard is not financial advice and forecasts are uncertain scenarios."


def apply_theme() -> None:
    st.set_page_config(page_title="AI Macro Market Intelligence", layout="wide", page_icon="📈")
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(circle at top left, #172554 0, #020617 28%, #030712 100%); color: #e5e7eb; }
        div[data-testid="stMetric"] { background: rgba(15,23,42,.78); border: 1px solid rgba(148,163,184,.22); border-radius: 8px; padding: 14px; }
        .info-card { background: rgba(15,23,42,.72); border: 1px solid rgba(56,189,248,.22); border-radius: 8px; padding: 16px; margin-bottom: 10px; }
        .risk-note { color: #fbbf24; font-size: .92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def db_ready() -> bool:
    return DUCKDB_PATH.exists()


@st.cache_data(show_spinner=False)
def load_table(table_name: str) -> pd.DataFrame:
    if not db_ready():
        return pd.DataFrame()
    return query_df(f"SELECT * FROM {table_name}")


@st.cache_data(show_spinner=False)
def run_query(sql: str) -> pd.DataFrame:
    if not db_ready():
        return pd.DataFrame()
    return query_df(sql)


def filters() -> dict:
    universe = read_asset_universe()
    sectors = ["All"] + sorted(universe.assets["sector"].dropna().unique().tolist())
    regions = ["Global", "Americas", "EMEA", "Asia"]
    with st.sidebar:
        st.caption(DISCLAIMER)
        selected_regions = st.multiselect("Region", regions, default=["Global"])
        selected_sectors = st.multiselect("Sector", sectors, default=["All"])
        benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "ACWI", "FEZ", "EWJ", "FXI", "EEM"], index=0)
        frequency = st.selectbox("Data frequency", ["1d", "1wk", "1mo"], index=0)
        horizon = st.slider("Forecast horizon", 1, 7, 3)
        model = st.selectbox(
            "Forecast model",
            ["Monte Carlo", "Historical average", "Moving average", "Exponential smoothing", "ARIMA", "Random forest"],
        )
        start_date = st.date_input("Start date", value=pd.Timestamp("2000-01-01"))
        end_date = st.date_input("End date", value=pd.Timestamp.today())
    return {
        "regions": selected_regions,
        "sectors": selected_sectors,
        "benchmark": benchmark,
        "frequency": frequency,
        "horizon": horizon,
        "model": model,
        "start_date": str(start_date),
        "end_date": str(end_date),
    }


def filter_assets(df: pd.DataFrame, selected: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if selected.get("regions") and "Global" not in selected["regions"] and "region" in out:
        out = out[out["region"].isin(selected["regions"])]
    if selected.get("sectors") and "All" not in selected["sectors"] and "sector" in out:
        out = out[out["sector"].isin(selected["sectors"])]
    return out


def empty_state() -> None:
    st.warning("No analytical DuckDB database found yet. Run a refresh from the Home page to create it.")


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)
