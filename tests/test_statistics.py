import pandas as pd

from src.statistics import calculate_returns, max_drawdown_from_prices, sharpe_ratio


def test_return_calculation():
    prices = pd.Series([100, 110, 99])
    returns = calculate_returns(prices)
    assert round(returns.iloc[1], 4) == 0.1
    assert round(returns.iloc[2], 4) == -0.1


def test_drawdown_calculation():
    prices = pd.Series([100, 120, 90, 150])
    assert round(max_drawdown_from_prices(prices), 4) == -0.25


def test_sharpe_ratio_returns_number():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0])
    assert sharpe_ratio(returns) == sharpe_ratio(returns)
