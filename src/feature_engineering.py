from __future__ import annotations

import numpy as np
import pandas as pd

from .statistics import (
    annualized_return,
    annualized_volatility,
    beta_and_corr,
    conditional_value_at_risk,
    cumulative_returns,
    max_drawdown_from_prices,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)


def date_dimension(dates: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(pd.Series(dates).drop_duplicates()).sort_values()
    return pd.DataFrame(
        {
            "date_key": d.dt.strftime("%Y%m%d").astype(int),
            "date_day": d.dt.date,
            "year": d.dt.year,
            "quarter": d.dt.quarter,
            "month": d.dt.month,
            "month_name": d.dt.month_name(),
            "week": d.dt.isocalendar().week.astype(int),
            "day": d.dt.day,
            "day_of_week": d.dt.day_name(),
            "is_weekend": d.dt.dayofweek >= 5,
        }
    )


def build_daily_returns(clean: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for asset_id, group in clean.groupby("asset_id"):
        g = group.sort_values("date").copy()
        ret = g["adjusted_close"].pct_change()
        g["daily_return"] = ret
        g["log_return"] = np.log(g["adjusted_close"] / g["adjusted_close"].shift(1))
        g["cumulative_return"] = cumulative_returns(ret)
        for window in [7, 30, 90]:
            g[f"rolling_return_{window}d"] = g["adjusted_close"].pct_change(window)
        for window in [30, 90]:
            g[f"rolling_volatility_{window}d"] = ret.rolling(window).std() * np.sqrt(252)
        frames.append(
            g[
                [
                    "asset_id",
                    "date",
                    "daily_return",
                    "log_return",
                    "cumulative_return",
                    "rolling_return_7d",
                    "rolling_return_30d",
                    "rolling_return_90d",
                    "rolling_volatility_30d",
                    "rolling_volatility_90d",
                ]
            ]
        )
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["date_key"] = pd.to_datetime(out["date"]).dt.strftime("%Y%m%d").astype(int)
        out = out.rename(columns={"date": "date_day"})
    return out


def _period_return(g: pd.DataFrame, trading_days: int) -> float:
    if len(g) <= trading_days:
        return np.nan
    return float(g["adjusted_close"].iloc[-1] / g["adjusted_close"].iloc[-trading_days] - 1)


def build_asset_features(clean: pd.DataFrame, benchmark_ticker: str = "SPY") -> pd.DataFrame:
    if clean.empty:
        return pd.DataFrame()
    benchmark = clean[clean["ticker"] == benchmark_ticker].sort_values("date")
    benchmark_returns = benchmark.set_index("date")["adjusted_close"].pct_change() if not benchmark.empty else pd.Series(dtype=float)
    rows = []
    for asset_id, group in clean.groupby("asset_id"):
        g = group.sort_values("date")
        returns = g.set_index("date")["adjusted_close"].pct_change()
        beta, corr = beta_and_corr(returns, benchmark_returns)
        rows.append(
            {
                "asset_id": asset_id,
                "calculation_date": g["date"].max().date(),
                "return_1m": _period_return(g, 21),
                "return_3m": _period_return(g, 63),
                "return_6m": _period_return(g, 126),
                "return_1y": _period_return(g, 252),
                "return_3y_annualized": annualized_return(returns.tail(252 * 3)) if len(g) >= 252 else np.nan,
                "annualized_return": annualized_return(returns),
                "annualized_volatility": annualized_volatility(returns),
                "sharpe_ratio": sharpe_ratio(returns),
                "sortino_ratio": sortino_ratio(returns),
                "max_drawdown": max_drawdown_from_prices(g["adjusted_close"]),
                "skewness": returns.skew(),
                "kurtosis": returns.kurt(),
                "var_95": value_at_risk(returns),
                "cvar_95": conditional_value_at_risk(returns),
                "beta_vs_benchmark": beta,
                "correlation_vs_benchmark": corr,
                "average_volume": g["volume"].mean(),
                "volume_volatility": g["volume"].pct_change().replace([np.inf, -np.inf], np.nan).std(),
            }
        )
    return pd.DataFrame(rows)


def build_monthly_returns(clean: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for asset_id, group in clean.groupby("asset_id"):
        monthly = (
            group.set_index("date")
            .sort_index()["adjusted_close"]
            .resample("MS")
            .last()
            .dropna()
            .pct_change()
            .rename("monthly_return")
            .to_frame()
        )
        monthly["asset_id"] = asset_id
        monthly["cumulative_monthly_return"] = cumulative_returns(monthly["monthly_return"])
        monthly["monthly_volatility"] = monthly["monthly_return"].rolling(12).std() * np.sqrt(12)
        frames.append(monthly.reset_index().rename(columns={"date": "month_start"}))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["month_key"] = pd.to_datetime(out["month_start"]).dt.strftime("%Y%m").astype(int)
    return out


def build_forecasting_input(clean: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for asset_id, group in clean.groupby("asset_id"):
        monthly = group.set_index("date").sort_index()["adjusted_close"].resample("MS").last().dropna().to_frame()
        monthly["asset_id"] = asset_id
        monthly["monthly_return"] = monthly["adjusted_close"].pct_change()
        monthly["rolling_mean"] = monthly["monthly_return"].rolling(12).mean()
        monthly["rolling_volatility"] = monthly["monthly_return"].rolling(12).std()
        monthly["momentum_3m"] = monthly["adjusted_close"].pct_change(3)
        monthly["momentum_6m"] = monthly["adjusted_close"].pct_change(6)
        monthly["momentum_12m"] = monthly["adjusted_close"].pct_change(12)
        frames.append(monthly.reset_index().rename(columns={"date": "date_day"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_group_performance(daily_returns: pd.DataFrame, assets: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if daily_returns.empty:
        return pd.DataFrame()
    df = daily_returns.merge(assets[["asset_id", group_col]], on="asset_id", how="left")
    grouped = df.groupby([group_col, "date_day"], as_index=False)["daily_return"].mean()
    grouped = grouped.rename(columns={group_col: group_col, "daily_return": "equal_weighted_return"})
    frames = []
    for key, group in grouped.groupby(group_col):
        g = group.sort_values("date_day").copy()
        g["cumulative_return"] = cumulative_returns(g["equal_weighted_return"])
        g["volatility"] = g["equal_weighted_return"].rolling(30).std() * np.sqrt(252)
        g["sharpe_ratio"] = g["equal_weighted_return"].rolling(252).mean() / g["equal_weighted_return"].rolling(252).std() * np.sqrt(252)
        g["max_drawdown"] = g["cumulative_return"].add(1).pipe(lambda x: x / x.cummax() - 1)
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
