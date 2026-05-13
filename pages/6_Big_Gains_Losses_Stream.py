from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.app_support import apply_theme, empty_state, filter_assets, filters, load_table
from src.plotting import style_figure
from src.storage import dataframe_csv_download


apply_theme()
st.title("Big Gains and Losses Stream")
selected = filters()
events = load_table("fact_detected_events")
assets = load_table("dim_asset")
if events.empty or assets.empty:
    empty_state()
    st.stop()

events = events.merge(assets[["asset_id", "ticker", "asset_name", "sector", "region"]], on="asset_id", how="left")
events = filter_assets(events, selected)

severity = st.multiselect("Severity", sorted(events["severity"].dropna().unique().tolist()), default=sorted(events["severity"].dropna().unique().tolist()))
event_type = st.multiselect("Event type", sorted(events["event_type"].dropna().unique().tolist()))
if severity:
    events = events[events["severity"].isin(severity)]
if event_type:
    events = events[events["event_type"].isin(event_type)]

summary = events.groupby("severity", as_index=False).size()
st.plotly_chart(style_figure(px.bar(summary, x="severity", y="size", color="severity", title="Severity summary")), use_container_width=True)
st.plotly_chart(style_figure(px.scatter(events, x="date_day", y="ticker", color="severity", size=events["event_value"].abs() + 0.01, hover_data=["event_type", "event_value", "price"], title="Event timeline")), use_container_width=True)

for row in events.sort_values("date_day", ascending=False).head(12).itertuples(index=False):
    st.markdown(
        f"<div class='info-card'><b>{row.ticker}</b> · {row.event_type} · {row.severity}<br>{row.explanation}<br>Value: {row.event_value:.2%} · Price: {row.price:.2f}</div>",
        unsafe_allow_html=True,
    )

st.dataframe(events.sort_values("date_day", ascending=False), use_container_width=True, hide_index=True)
st.download_button("Export events CSV", dataframe_csv_download(events), "detected_events.csv", "text/csv")
