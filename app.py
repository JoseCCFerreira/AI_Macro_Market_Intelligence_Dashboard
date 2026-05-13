from __future__ import annotations

import pandas as pd
import streamlit as st

from src import relational_db
from src.app_support import (
    DISCLAIMER,
    apply_theme,
    db_ready,
    explain_card,
    filter_assets,
    filters,
    load_table,
    metric_row,
    performance_narrative,
)
from src.asset_universe import read_asset_universe
from src.etl_pipeline import run_full_refresh
from src.plotting import performance_bar, risk_return_scatter
from src.storage import dataframe_csv_download
from src.utils import SQLITE_PATH


apply_theme()

st.title("AI Macro Market Intelligence Dashboard")
st.caption(DISCLAIMER)

selected = filters()

with st.sidebar:
    st.divider()
    tickers_df = read_asset_universe().assets
    available_tickers = tickers_df["ticker"].tolist()
    selected_tickers = st.multiselect("Tickers", available_tickers, default=available_tickers[:12])
    if st.button("Refresh market data", type="primary", use_container_width=True):
        st.cache_data.clear()
        with st.spinner("Running yfinance -> SQLite -> quality -> DuckDB refresh..."):
            status = run_full_refresh(
                start_date=selected["start_date"],
                end_date=selected["end_date"],
                frequency=selected["frequency"],
                selected_tickers=selected_tickers or None,
                benchmark_ticker=selected["benchmark"],
            )
        st.session_state["last_refresh_status"] = status.to_dict()
        st.rerun()
    if st.button("Clear Streamlit cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared.")

st.markdown(
    """
    <div class="info-card">
    <strong>What this is:</strong> a local market intelligence platform. Data is downloaded from yfinance,
    stored first in SQLite for traceability, checked for quality, transformed into DuckDB analytical marts,
    and only then consumed by Streamlit.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <p class="risk-note">
    Read this like an analyst, not like a fortune teller: strong past performance can reverse, low volatility can rise,
    and every forecast is a scenario with uncertainty.
    </p>
    """,
    unsafe_allow_html=True,
)

status = st.session_state.get("last_refresh_status")
if status:
    st.subheader("Latest refresh status")
    metric_row(
        [
            ("Status", status["status"], None),
            ("Requested", str(status["requested_tickers"]), None),
            ("Loaded", str(status["successful_tickers"]), None),
            ("Failed", str(status["failed_tickers"]), None),
            ("SQLite rows", f"{status['rows_inserted_sqlite']:,}", None),
            ("DuckDB rows", f"{status['rows_loaded_duckdb']:,}", None),
        ]
    )
    if status.get("failed_symbols"):
        st.warning("Failed tickers: " + ", ".join(status["failed_symbols"]))
    st.caption(status.get("message", ""))

if not db_ready():
    st.info("Click Refresh market data in the sidebar to create the SQLite and DuckDB databases.")
    st.stop()

perf = filter_assets(load_table("mart_asset_performance"), selected)
features = filter_assets(load_table("mart_clustering_input"), selected)
events = load_table("fact_detected_events")

if perf.empty:
    st.warning("DuckDB exists, but no performance mart is populated. Run a refresh.")
    st.stop()

best_ytd = perf.sort_values("ytd_return", ascending=False).head(1)
worst_ytd = perf.sort_values("ytd_return").head(1)
highest_vol = perf.sort_values("annualized_volatility", ascending=False).head(1)
deepest_dd = perf.sort_values("max_drawdown").head(1)

metric_row(
    [
        ("Assets loaded", f"{perf['ticker'].nunique():,}", None),
        ("Best YTD", best_ytd["ticker"].iloc[0], f"{best_ytd['ytd_return'].iloc[0]:.2%}"),
        ("Worst YTD", worst_ytd["ticker"].iloc[0], f"{worst_ytd['ytd_return'].iloc[0]:.2%}"),
        ("Highest volatility", highest_vol["ticker"].iloc[0], f"{highest_vol['annualized_volatility'].iloc[0]:.2%}"),
        ("Max drawdown", deepest_dd["ticker"].iloc[0], f"{deepest_dd['max_drawdown'].iloc[0]:.2%}"),
    ]
)

st.markdown(f"<div class='analysis-card'><strong>Current read:</strong><br>{performance_narrative(perf)}</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Risk Map", "Analysis Guide", "Quality & Audit"])
with tab1:
    col1, col2 = st.columns(2)
    sector_perf = perf.groupby("sector", as_index=False)["ytd_return"].mean(numeric_only=True)
    region_perf = perf.groupby("region", as_index=False)["ytd_return"].mean(numeric_only=True)
    col1.plotly_chart(performance_bar(sector_perf, "sector", "ytd_return", "Average YTD return by sector"), use_container_width=True)
    col2.plotly_chart(performance_bar(region_perf, "region", "ytd_return", "Average YTD return by region"), use_container_width=True)
    st.dataframe(perf.sort_values("ytd_return", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Export performance CSV", dataframe_csv_download(perf), "asset_performance.csv", "text/csv")

with tab2:
    st.plotly_chart(risk_return_scatter(features), use_container_width=True)
    explain_card(
        "How to read the risk map",
        "Assets higher on the chart have stronger annualized return. Assets further right have more volatility. "
        "The top-right corner can look attractive, but it usually means higher uncertainty and bigger potential drawdowns.",
    )

with tab3:
    col1, col2 = st.columns(2)
    winners = perf.sort_values("ytd_return", ascending=False).head(8)
    losers = perf.sort_values("ytd_return").head(8)
    risk = perf.sort_values("annualized_volatility", ascending=False).head(8)
    defensive = perf.sort_values("annualized_volatility").head(8)
    col1.subheader("Momentum leaders")
    col1.dataframe(winners[["ticker", "sector", "region", "ytd_return", "annualized_volatility", "sharpe_ratio"]], use_container_width=True, hide_index=True)
    col2.subheader("Weakest YTD")
    col2.dataframe(losers[["ticker", "sector", "region", "ytd_return", "max_drawdown", "sharpe_ratio"]], use_container_width=True, hide_index=True)
    col3, col4 = st.columns(2)
    col3.subheader("Highest risk")
    col3.dataframe(risk[["ticker", "sector", "region", "annualized_volatility", "max_drawdown", "return_1y"]], use_container_width=True, hide_index=True)
    col4.subheader("Lower volatility")
    col4.dataframe(defensive[["ticker", "sector", "region", "annualized_volatility", "max_drawdown", "return_1y"]], use_container_width=True, hide_index=True)
    explain_card(
        "Simple interpretation",
        "Momentum leaders are assets that have been performing well recently. Weak assets may be in a correction or structural decline. "
        "High volatility assets can move fast in both directions. Lower volatility assets are calmer, but not automatically safer.",
    )

with tab4:
    st.write("Relational staging database:", str(SQLITE_PATH))
    try:
        with relational_db.connect(SQLITE_PATH) as conn:
            logs = relational_db.read_quality_logs(conn)
        st.dataframe(logs, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.info(f"No quality logs available yet: {exc}")
    st.write("Recent detected events")
    st.dataframe(events.tail(50), use_container_width=True, hide_index=True)
