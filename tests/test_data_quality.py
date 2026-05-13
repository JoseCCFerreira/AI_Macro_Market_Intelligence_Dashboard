import pandas as pd

from src.data_quality import validate_market_data


def test_data_quality_detects_invalid_prices():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": ["2024-01-01", "2024-01-02"],
            "open": [10, 0],
            "high": [11, 0],
            "low": [9, 0],
            "close": [10, 0],
            "adjusted_close": [10, 0],
            "volume": [1000, None],
        }
    )
    issues = validate_market_data(df)
    assert "non_positive_close" in issues["issue_type"].tolist()
    assert not issues.empty
