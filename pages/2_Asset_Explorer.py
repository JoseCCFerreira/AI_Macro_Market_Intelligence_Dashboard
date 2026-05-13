from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_support import apply_theme, empty_state, filters, load_table, run_query
from src.plotting import drawdown_chart, style_figure
from src.statistics import descriptive_statistics
from src.storage import dataframe_csv_download


apply_theme()
st.title("Asset Explorer")
filters()

assets = load_table("dim_asset")
if assets.empty:
    empty_state()
    st.stop()

ticker = st.selectbox("Asset", assets["ticker"].sort_values().tolist())
prices = run_query(
    """
    SELECT p.date_day, p.adjusted_close, p.volume, r.daily_return, r.cumulative_return, a.ticker
    FROM fact_market_prices p
    JOIN fact_daily_returns r USING(asset_id, date_day)
    JOIN dim_asset a USING(asset_id)
    WHERE a.ticker = ?
    ORDER BY p.date_day
    """.replace("?", f"'{ticker}'")
)

if prices.empty:
    st.warning("No prices found for this asset.")
    st.stop()

prices["ma20"] = prices["adjusted_close"].rolling(20).mean()
prices["ma50"] = prices["adjusted_close"].rolling(50).mean()
prices["ma100"] = prices["adjusted_close"].rolling(100).mean()
prices["ma200"] = prices["adjusted_close"].rolling(200).mean()
prices["rolling_volatility"] = prices["daily_return"].rolling(30).std() * (252**0.5)
prices["rolling_sharpe"] = prices["daily_return"].rolling(126).mean() / prices["daily_return"].rolling(126).std() * (252**0.5)
prices["drawdown"] = prices["adjusted_close"] / prices["adjusted_close"].cummax() - 1

fig = px.line(prices, x="date_day", y=["adjusted_close", "ma20", "ma50", "ma100", "ma200"], title=f"{ticker} price and moving averages")
st.plotly_chart(style_figure(fig), use_container_width=True)

col1, col2 = st.columns(2)
col1.plotly_chart(style_figure(px.line(prices, x="date_day", y="rolling_volatility", title="Rolling volatility")), use_container_width=True)
col2.plotly_chart(drawdown_chart(prices, "Maximum drawdown path"), use_container_width=True)

col3, col4 = st.columns(2)
col3.plotly_chart(style_figure(px.histogram(prices.dropna(), x="daily_return", nbins=80, title="Return distribution")), use_container_width=True)
col4.plotly_chart(style_figure(px.bar(prices.tail(365), x="date_day", y="volume", title="Volume evolution")), use_container_width=True)

stats = descriptive_statistics(prices["daily_return"], prices["adjusted_close"])
periods = {"ytd_return": prices[prices["date_day"].astype(str).str[:4] == str(pd.Timestamp.today().year)], "1y": 252, "3y": 252 * 3, "5y": 252 * 5, "10y": 252 * 10}
extra = {}
for label, days in list(periods.items())[1:]:
    extra[f"return_{label}"] = prices["adjusted_close"].iloc[-1] / prices["adjusted_close"].iloc[-days] - 1 if len(prices) > days else None
stats_df = pd.DataFrame([{**stats, **extra}]).T.reset_index()
stats_df.columns = ["metric", "value"]
st.dataframe(stats_df, use_container_width=True, hide_index=True)
st.download_button("Export selected asset CSV", dataframe_csv_download(prices), f"{ticker}_history.csv", "text/csv")
