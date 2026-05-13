import pandas as pd

from src.clustering import prepare_feature_matrix, run_kmeans


def _features():
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "sector": ["AI", "AI", "Energy", "Metals"],
            "region": ["Americas", "EMEA", "Asia", "Global"],
            "return_1m": [0.1, 0.05, -0.01, 0.02],
            "return_3m": [0.2, 0.1, 0.0, 0.03],
            "return_6m": [0.3, 0.12, 0.01, 0.04],
            "return_1y": [0.4, 0.15, 0.02, 0.05],
            "return_3y_annualized": [0.2, 0.1, 0.04, 0.03],
            "annualized_volatility": [0.5, 0.3, 0.2, 0.15],
            "sharpe_ratio": [1.0, 0.8, 0.2, 0.4],
            "max_drawdown": [-0.4, -0.2, -0.1, -0.08],
            "skewness": [0, 0.1, -0.1, 0.05],
            "kurtosis": [3, 2, 1, 1.5],
            "average_volume": [1000, 800, 600, 500],
            "beta_vs_benchmark": [1.5, 1.2, 0.8, 0.2],
            "correlation_vs_benchmark": [0.8, 0.7, 0.5, 0.1],
        }
    )


def test_prepare_feature_matrix():
    clean, matrix, _ = prepare_feature_matrix(_features())
    assert clean.shape[0] == 4
    assert matrix.shape[0] == 4


def test_kmeans_assigns_clusters():
    result = run_kmeans(_features(), n_clusters=2)
    assert "cluster" in result["assignments"].columns
    assert result["assignments"]["cluster"].nunique() == 2
