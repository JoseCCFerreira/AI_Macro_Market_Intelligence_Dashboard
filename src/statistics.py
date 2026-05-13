from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera


TRADING_DAYS = 252


def calculate_returns(prices: pd.Series) -> pd.Series:
    return prices.astype(float).pct_change()


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod() - 1


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    return float((1 + r).prod() ** (periods / len(r)) - 1)


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.dropna().std() * np.sqrt(periods)) if returns.dropna().size > 1 else np.nan


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    r = returns.dropna()
    vol = annualized_volatility(r)
    if not np.isfinite(vol) or vol == 0:
        return np.nan
    return float((annualized_return(r) - risk_free_rate) / vol)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    r = returns.dropna()
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    if not np.isfinite(downside) or downside == 0:
        return np.nan
    return float((annualized_return(r) - risk_free_rate) / downside)


def max_drawdown_from_prices(prices: pd.Series) -> float:
    p = prices.dropna().astype(float)
    if p.empty:
        return np.nan
    drawdown = p / p.cummax() - 1
    return float(drawdown.min())


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    r = returns.dropna()
    return float(r.quantile(1 - level)) if not r.empty else np.nan


def conditional_value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    var = value_at_risk(r, level)
    tail = r[r <= var]
    return float(tail.mean()) if not tail.empty else np.nan


def beta_and_corr(asset_returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 3:
        return np.nan, np.nan
    x = aligned.iloc[:, 1]
    y = aligned.iloc[:, 0]
    variance = x.var()
    beta = np.nan if variance == 0 else y.cov(x) / variance
    return float(beta), float(y.corr(x))


def descriptive_statistics(returns: pd.Series, prices: pd.Series | None = None) -> dict:
    r = returns.dropna()
    p = prices.dropna() if prices is not None else pd.Series(dtype=float)
    normality_p = np.nan
    autocorr_p = np.nan
    if len(r) >= 8:
        normality_p = float(jarque_bera(r).pvalue)
    if len(r) >= 20:
        autocorr_p = float(acorr_ljungbox(r, lags=[min(10, len(r) // 2)], return_df=True)["lb_pvalue"].iloc[0])
    return {
        "mean": float(r.mean()) if not r.empty else np.nan,
        "median": float(r.median()) if not r.empty else np.nan,
        "annualized_return": annualized_return(r),
        "annualized_volatility": annualized_volatility(r),
        "sharpe_ratio": sharpe_ratio(r),
        "sortino_ratio": sortino_ratio(r),
        "max_drawdown": max_drawdown_from_prices(p) if not p.empty else np.nan,
        "skewness": float(r.skew()) if len(r) > 2 else np.nan,
        "kurtosis": float(r.kurt()) if len(r) > 3 else np.nan,
        "var_95": value_at_risk(r),
        "cvar_95": conditional_value_at_risk(r),
        "normality_p_value": normality_p,
        "autocorrelation_p_value": autocorr_p,
    }
