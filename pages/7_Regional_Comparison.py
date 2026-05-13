from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.app_support import apply_theme, empty_state, filters, load_table
from src.plotting import style_figure
from src.storage import dataframe_csv_download


apply_theme()
st.title("Regional Comparison")
filters()

regional = load_table("mart_regional_performance")
perf = load_table("mart_asset_performance")
if regional.empty or perf.empty:
    empty_state()
    st.stop()

st.plotly_chart(style_figure(px.line(regional, x="date_day", y="cumulative_return", color="region", title="Equal-weighted regional cumulative returns")), use_container_width=True)

summary = perf.groupby("region", as_index=False)[["annualized_volatility", "sharpe_ratio", "max_drawdown", "ytd_return"]].mean(numeric_only=True)
st.dataframe(summary, use_container_width=True, hide_index=True)

sector_region = perf.groupby(["region", "sector"], as_index=False)["ytd_return"].mean(numeric_only=True)
st.plotly_chart(style_figure(px.bar(sector_region, x="region", y="ytd_return", color="sector", barmode="group", title="Best and worst sectors by region")), use_container_width=True)

for region in sorted(perf["region"].dropna().unique()):
    st.subheader(region)
    col1, col2 = st.columns(2)
    col1.write("Top 10 winners")
    col1.dataframe(perf[perf["region"] == region].sort_values("ytd_return", ascending=False).head(10), use_container_width=True, hide_index=True)
    col2.write("Top 10 losers")
    col2.dataframe(perf[perf["region"] == region].sort_values("ytd_return").head(10), use_container_width=True, hide_index=True)

st.download_button("Export regional performance CSV", dataframe_csv_download(regional), "regional_performance.csv", "text/csv")
