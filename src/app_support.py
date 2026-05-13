from __future__ import annotations

import pandas as pd
import streamlit as st

from .analytics_db import query_df
from .asset_universe import read_asset_universe
from .utils import DUCKDB_PATH


DISCLAIMER = "Educational analytics only. This dashboard is not financial advice and forecasts are uncertain scenarios."


def apply_theme() -> None:
    st.set_page_config(page_title="AI Macro Market Intelligence", layout="wide", page_icon="AI")
    st.markdown(
        """
        <style>
        :root {
            --bg: #f8fafc;
            --panel: #ffffff;
            --panel-soft: #eef2ff;
            --text: #0f172a;
            --muted: #475569;
            --line: #cbd5e1;
            --blue: #1d4ed8;
            --green: #047857;
            --red: #b91c1c;
            --amber: #b45309;
        }
        .stApp {
            background:
                linear-gradient(135deg, rgba(219,234,254,.92) 0%, rgba(248,250,252,.98) 34%, rgba(236,253,245,.92) 100%);
            color: var(--text);
        }
        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid rgba(255,255,255,.14);
        }
        section[data-testid="stSidebar"] * { color: #f8fafc !important; }
        h1, h2, h3, h4, p, label, span, div { letter-spacing: 0; }
        h1 { color: #0f172a; font-weight: 800; }
        h2, h3 { color: #1e293b; }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.94);
            border: 1px solid rgba(30,64,175,.18);
            border-left: 5px solid #2563eb;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 8px 20px rgba(15,23,42,.08);
        }
        div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0f172a !important;
        }
        div[data-testid="stMetricDelta"] { color: #047857 !important; }
        .info-card, .analysis-card, .explain-card {
            background: rgba(255,255,255,.96);
            border: 1px solid rgba(148,163,184,.48);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 10px 24px rgba(15,23,42,.07);
            color: #0f172a;
        }
        .analysis-card strong { color: #1d4ed8; }
        .risk-note {
            background: #fffbeb;
            border: 1px solid #f59e0b;
            border-left: 5px solid #d97706;
            border-radius: 8px;
            color: #78350f;
            padding: 12px 14px;
            font-size: .94rem;
        }
        .good { color: var(--green); font-weight: 700; }
        .bad { color: var(--red); font-weight: 700; }
        .warn { color: var(--amber); font-weight: 700; }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148,163,184,.45);
            border-radius: 8px;
            overflow: hidden;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,.88);
            border: 1px solid rgba(148,163,184,.55);
            border-radius: 8px 8px 0 0;
            color: #0f172a;
            font-weight: 700;
        }
        .stTabs [aria-selected="true"] {
            background: #1d4ed8;
            color: #ffffff;
        }
        a { color: #1d4ed8; font-weight: 700; }
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


def explain_card(title: str, body: str) -> None:
    st.markdown(f"<div class='explain-card'><strong>{title}</strong><br>{body}</div>", unsafe_allow_html=True)


def performance_narrative(perf: pd.DataFrame) -> str:
    if perf.empty:
        return "No performance data is available yet."
    best = perf.sort_values("ytd_return", ascending=False).iloc[0]
    worst = perf.sort_values("ytd_return").iloc[0]
    risky = perf.sort_values("annualized_volatility", ascending=False).iloc[0]
    drawdown = perf.sort_values("max_drawdown").iloc[0]
    return (
        f"<span class='good'>{best['ticker']}</span> is the strongest YTD asset in the current filter "
        f"({best['ytd_return']:.2%}), while <span class='bad'>{worst['ticker']}</span> is the weakest "
        f"({worst['ytd_return']:.2%}). The highest volatility asset is "
        f"<span class='warn'>{risky['ticker']}</span> ({risky['annualized_volatility']:.2%}), and the deepest "
        f"historical drawdown belongs to <span class='bad'>{drawdown['ticker']}</span> ({drawdown['max_drawdown']:.2%})."
    )
