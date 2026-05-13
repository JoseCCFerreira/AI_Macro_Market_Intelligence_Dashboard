from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.app_support import apply_theme, empty_state, filter_assets, filters, load_table, run_query
from src.plotting import correlation_heatmap, risk_return_scatter, style_figure
from src.storage import dataframe_csv_download


apply_theme()
st.title("Statistical Analysis")
selected = filters()

features = filter_assets(load_table("mart_clustering_input"), selected)
daily = run_query("SELECT r.*, a.ticker, a.sector, a.region FROM fact_daily_returns r JOIN dim_asset a USING(asset_id)")
daily = filter_assets(daily, selected)

if features.empty or daily.empty:
    empty_state()
    st.stop()

st.markdown("Normality tests, autocorrelation and PCA help describe behavior. They do not prove future returns.")

returns = daily.pivot_table(index="date_day", columns="ticker", values="daily_return").dropna(axis=1, thresh=60)
col1, col2 = st.columns(2)
if not returns.empty:
    col1.plotly_chart(correlation_heatmap(returns.tail(756).corr(), "Three-year correlation matrix"), use_container_width=True)
    sample = returns.iloc[:, : min(8, returns.shape[1])].melt(value_name="daily_return")
    col2.plotly_chart(style_figure(px.histogram(sample.dropna(), x="daily_return", color="ticker", nbins=70, title="Return distribution")), use_container_width=True)

st.plotly_chart(risk_return_scatter(features, "Risk-return scatter by sector"), use_container_width=True)

numeric = features.select_dtypes("number").replace([float("inf"), float("-inf")], pd.NA).fillna(0)
if len(numeric) >= 3 and numeric.shape[1] >= 2:
    scaled = StandardScaler().fit_transform(numeric)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    pca_df = features[["ticker", "sector", "region"]].copy()
    pca_df["pca_1"] = coords[:, 0]
    pca_df["pca_2"] = coords[:, 1]
    st.plotly_chart(style_figure(px.scatter(pca_df, x="pca_1", y="pca_2", color="sector", symbol="region", hover_name="ticker", title="PCA dimensionality reduction")), use_container_width=True)

sector = features.groupby("sector", as_index=False)[["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]].mean(numeric_only=True)
region = features.groupby("region", as_index=False)[["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]].mean(numeric_only=True)
st.dataframe(sector, use_container_width=True, hide_index=True)
st.dataframe(region, use_container_width=True, hide_index=True)
st.download_button("Export statistics CSV", dataframe_csv_download(features), "statistics.csv", "text/csv")
