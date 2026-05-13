from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "return_1m",
    "return_3m",
    "return_6m",
    "return_1y",
    "return_3y_annualized",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "skewness",
    "kurtosis",
    "average_volume",
    "beta_vs_benchmark",
    "correlation_vs_benchmark",
]


def prepare_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not available:
        raise ValueError("No clustering features available.")
    clean = df.copy()
    clean[available] = clean[available].replace([np.inf, -np.inf], np.nan)
    clean[available] = clean[available].fillna(clean[available].median(numeric_only=True)).fillna(0)
    scaler = StandardScaler()
    matrix = scaler.fit_transform(clean[available])
    return clean, matrix, scaler


def run_kmeans(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    clean, matrix, _ = prepare_feature_matrix(df)
    n_clusters = max(2, min(n_clusters, len(clean)))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = model.fit_predict(matrix)
    return _cluster_output(clean, matrix, labels, "kmeans", inertia=float(model.inertia_))


def run_agglomerative(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    clean, matrix, _ = prepare_feature_matrix(df)
    n_clusters = max(2, min(n_clusters, len(clean)))
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(matrix)
    return _cluster_output(clean, matrix, labels, "agglomerative")


def _cluster_output(clean: pd.DataFrame, matrix: np.ndarray, labels: np.ndarray, method: str, inertia: float | None = None) -> dict:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)
    assignments = clean.copy()
    assignments["cluster"] = labels
    assignments["pca_1"] = coords[:, 0]
    assignments["pca_2"] = coords[:, 1]
    silhouette = silhouette_score(matrix, labels) if len(set(labels)) > 1 and len(labels) > len(set(labels)) else np.nan
    profiles = assignments.groupby("cluster")[FEATURE_COLUMNS].mean(numeric_only=True).reset_index()
    profiles["cluster_size"] = assignments.groupby("cluster").size().values
    profiles["interpretation"] = profiles.apply(_interpret_cluster, axis=1)
    return {
        "method": method,
        "assignments": assignments,
        "profiles": profiles,
        "silhouette_score": float(silhouette) if np.isfinite(silhouette) else np.nan,
        "inertia": inertia,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    }


def _interpret_cluster(row: pd.Series) -> str:
    if row.get("annualized_volatility", 0) > 0.4 and row.get("return_1y", 0) > 0.15:
        return "High growth / high volatility"
    if row.get("max_drawdown", 0) < -0.35:
        return "Deep drawdown / cyclical risk"
    if row.get("annualized_volatility", 0) < 0.2:
        return "Defensive / lower volatility"
    if row.get("correlation_vs_benchmark", 0) < 0.25:
        return "Diversifier / hedge-like behavior"
    return "Market-sensitive balanced profile"
