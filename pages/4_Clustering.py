from __future__ import annotations

import streamlit as st

from src.app_support import apply_theme, empty_state, filter_assets, filters, load_table
from src.clustering import run_agglomerative, run_kmeans
from src.plotting import cluster_pca_chart
from src.storage import dataframe_csv_download


apply_theme()
st.title("Clustering")
selected = filters()
features = filter_assets(load_table("mart_clustering_input"), selected)

if features.empty:
    empty_state()
    st.stop()

method = st.radio("Method", ["KMeans", "Agglomerative"], horizontal=True)
n_clusters = st.slider("Number of clusters", 2, min(10, max(2, len(features))), 4)

result = run_kmeans(features, n_clusters) if method == "KMeans" else run_agglomerative(features, n_clusters)

col1, col2 = st.columns(2)
col1.metric("Silhouette score", f"{result['silhouette_score']:.3f}" if result["silhouette_score"] == result["silhouette_score"] else "n/a")
col2.metric("Inertia", f"{result['inertia']:.2f}" if result["inertia"] is not None else "n/a")

st.plotly_chart(cluster_pca_chart(result["assignments"]), use_container_width=True)
st.subheader("Cluster profiles")
st.dataframe(result["profiles"], use_container_width=True, hide_index=True)
st.subheader("Assignments")
st.dataframe(result["assignments"][["ticker", "sector", "region", "cluster", "pca_1", "pca_2"]], use_container_width=True, hide_index=True)
st.download_button("Export cluster assignments CSV", dataframe_csv_download(result["assignments"]), "cluster_assignments.csv", "text/csv")
