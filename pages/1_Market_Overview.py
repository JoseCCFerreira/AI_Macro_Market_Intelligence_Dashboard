from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_support import apply_theme, empty_state, filter_assets, filters, load_table, metric_row, run_query
from src.plotting import correlation_heatmap, cumulative_return_chart, performance_bar
from src.storage import dataframe_csv_download


apply_theme()
st.title("Market Overview")
selected = filters()

perf = filter_assets(load_table("mart_asset_performance"), selected)
daily = run_query(
    """
    SELECT r.*, a.ticker, a.sector, a.region
    FROM fact_daily_returns r JOIN dim_asset a USING(asset_id)
    """
)
daily = filter_assets(daily, selected)

if perf.empty or daily.empty:
    empty_state()
    st.stop()

latest = daily.sort_values("date_day").groupby("ticker").tail(1)
best_today = latest.sort_values("daily_return", ascending=False).head(1)
worst_today = latest.sort_values("daily_return").head(1)

metric_row(
    [
        ("Assets", f"{perf['ticker'].nunique():,}", None),
        ("Best today", best_today["ticker"].iloc[0], f"{best_today['daily_return'].iloc[0]:.2%}"),
        ("Worst today", worst_today["ticker"].iloc[0], f"{worst_today['daily_return'].iloc[0]:.2%}"),
        ("Avg volatility", f"{perf['annualized_volatility'].mean():.2%}", None),
    ]
)

col1, col2 = st.columns(2)
sector_perf = perf.groupby("sector", as_index=False)["ytd_return"].mean(numeric_only=True)
regional_perf = perf.groupby("region", as_index=False)["ytd_return"].mean(numeric_only=True)
col1.plotly_chart(performance_bar(sector_perf, "sector", "ytd_return", "Sector performance comparison"), use_container_width=True)
col2.plotly_chart(performance_bar(regional_perf, "region", "ytd_return", "Regional performance comparison"), use_container_width=True)

top_tickers = perf.sort_values("ytd_return", ascending=False)["ticker"].head(10).tolist()
chart_df = daily[daily["ticker"].isin(top_tickers)]
st.plotly_chart(cumulative_return_chart(chart_df, "Top assets cumulative returns"), use_container_width=True)

returns = daily.pivot_table(index="date_day", columns="ticker", values="daily_return").dropna(axis=1, thresh=30)
if not returns.empty:
    st.plotly_chart(correlation_heatmap(returns.tail(252).corr(), "One-year correlation heatmap"), use_container_width=True)

volume = run_query(
    """
    SELECT p.date_day, a.ticker, p.volume
    FROM fact_market_prices p JOIN dim_asset a USING(asset_id)
    ORDER BY p.date_day
    """
)
volume = filter_assets(volume, selected)
if not volume.empty:
    st.line_chart(volume.pivot_table(index="date_day", columns="ticker", values="volume").tail(365))

st.download_button("Export overview CSV", dataframe_csv_download(perf), "market_overview.csv", "text/csv")
