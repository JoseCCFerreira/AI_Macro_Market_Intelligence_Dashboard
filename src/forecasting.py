from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def prepare_monthly_forecasting_dataset(asset_prices: pd.DataFrame) -> pd.DataFrame:
    if asset_prices.empty:
        return pd.DataFrame()
    df = asset_prices.copy()
    df["date_day"] = pd.to_datetime(df.get("date_day", df.get("date")))
    monthly = df.set_index("date_day").sort_index()["adjusted_close"].resample("MS").last().dropna().to_frame()
    monthly["monthly_return"] = monthly["adjusted_close"].pct_change()
    monthly["lag_1"] = monthly["monthly_return"].shift(1)
    monthly["lag_3"] = monthly["monthly_return"].shift(3)
    monthly["rolling_mean"] = monthly["monthly_return"].rolling(12).mean()
    monthly["rolling_volatility"] = monthly["monthly_return"].rolling(12).std()
    monthly["momentum_6m"] = monthly["adjusted_close"].pct_change(6)
    return monthly.reset_index().rename(columns={"date_day": "month_start"})


def forecast_asset(monthly: pd.DataFrame, horizon_years: int = 3, model_name: str = "Monte Carlo") -> dict:
    if monthly.empty or monthly["adjusted_close"].dropna().shape[0] < 18:
        return {"forecast": pd.DataFrame(), "metrics": {"error": "Insufficient history"}, "model": model_name}
    horizon_months = int(max(1, min(7, horizon_years)) * 12)
    model_name = model_name.lower()
    if "historical" in model_name:
        return historical_average_forecast(monthly, horizon_months)
    if "moving" in model_name:
        return moving_average_forecast(monthly, horizon_months)
    if "exponential" in model_name:
        return exponential_smoothing_forecast(monthly, horizon_months)
    if "arima" in model_name:
        return arima_forecast(monthly, horizon_months)
    if "random" in model_name:
        return ml_regression_forecast(monthly, horizon_months)
    return monte_carlo_forecast(monthly, horizon_months)


def _future_index(last_date: pd.Timestamp, horizon_months: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(last_date) + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    aligned = pd.concat([actual, predicted], axis=1).dropna()
    if aligned.empty:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan}
    mae = mean_absolute_error(aligned.iloc[:, 0], aligned.iloc[:, 1])
    rmse = mean_squared_error(aligned.iloc[:, 0], aligned.iloc[:, 1]) ** 0.5
    denom = aligned.iloc[:, 0].replace(0, np.nan).abs()
    mape = ((aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs() / denom).mean()
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape) if np.isfinite(mape) else np.nan}


def historical_average_forecast(monthly: pd.DataFrame, horizon_months: int) -> dict:
    returns = monthly["monthly_return"].dropna()
    avg = returns.mean()
    vol = returns.std()
    return _path_from_return_assumption(monthly, horizon_months, avg, vol, "Historical average return")


def moving_average_forecast(monthly: pd.DataFrame, horizon_months: int) -> dict:
    returns = monthly["monthly_return"].dropna()
    avg = returns.tail(12).mean()
    vol = returns.tail(36).std()
    return _path_from_return_assumption(monthly, horizon_months, avg, vol, "Moving average trend")


def _path_from_return_assumption(monthly: pd.DataFrame, horizon_months: int, avg: float, vol: float, model: str) -> dict:
    last_price = monthly["adjusted_close"].dropna().iloc[-1]
    dates = _future_index(monthly["month_start"].iloc[-1], horizon_months)
    steps = np.arange(1, horizon_months + 1)
    median = last_price * (1 + avg) ** steps
    band = 1.96 * (vol or 0) * np.sqrt(steps)
    forecast = pd.DataFrame(
        {
            "date_day": dates,
            "forecast": median,
            "lower": median * (1 - band),
            "upper": median * (1 + band),
            "scenario": "base",
        }
    )
    backtest = monthly["adjusted_close"].shift(1) * (1 + avg)
    return {"forecast": forecast, "metrics": _metrics(monthly["adjusted_close"], backtest), "model": model}


def exponential_smoothing_forecast(monthly: pd.DataFrame, horizon_months: int) -> dict:
    try:
        series = monthly.set_index("month_start")["adjusted_close"].asfreq("MS").ffill()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ExponentialSmoothing(series, trend="add", seasonal=None).fit(optimized=True)
        pred = fitted.forecast(horizon_months)
        resid_std = fitted.resid.std()
        forecast = pd.DataFrame({"date_day": pred.index, "forecast": pred.values})
        forecast["lower"] = forecast["forecast"] - 1.96 * resid_std
        forecast["upper"] = forecast["forecast"] + 1.96 * resid_std
        forecast["scenario"] = "base"
        return {"forecast": forecast, "metrics": _metrics(series, fitted.fittedvalues), "model": "Exponential smoothing"}
    except Exception as exc:
        return {"forecast": pd.DataFrame(), "metrics": {"error": str(exc)}, "model": "Exponential smoothing"}


def arima_forecast(monthly: pd.DataFrame, horizon_months: int) -> dict:
    try:
        series = monthly.set_index("month_start")["adjusted_close"].asfreq("MS").ffill()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(series, order=(1, 1, 1)).fit()
        pred = fitted.get_forecast(horizon_months)
        frame = pred.summary_frame(alpha=0.1)
        forecast = pd.DataFrame(
            {
                "date_day": frame.index,
                "forecast": frame["mean"].values,
                "lower": frame.iloc[:, 2].values,
                "upper": frame.iloc[:, 3].values,
                "scenario": "base",
            }
        )
        return {"forecast": forecast, "metrics": {"aic": float(fitted.aic)}, "model": "ARIMA(1,1,1)"}
    except Exception as exc:
        return {"forecast": pd.DataFrame(), "metrics": {"error": str(exc)}, "model": "ARIMA"}


def ml_regression_forecast(monthly: pd.DataFrame, horizon_months: int) -> dict:
    df = monthly.dropna().copy()
    features = ["lag_1", "lag_3", "rolling_mean", "rolling_volatility", "momentum_6m"]
    if len(df) < 24:
        return historical_average_forecast(monthly, horizon_months)
    model = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=3)
    split = max(12, int(len(df) * 0.8))
    model.fit(df[features].iloc[:split], df["monthly_return"].iloc[:split])
    pred_test = pd.Series(model.predict(df[features].iloc[split:]), index=df.index[split:])
    metrics = _metrics(df["monthly_return"].iloc[split:], pred_test)
    returns = []
    state = df.iloc[-1:].copy()
    for _ in range(horizon_months):
        predicted_return = float(model.predict(state[features])[0])
        returns.append(predicted_return)
        state.loc[:, "lag_3"] = state["lag_1"]
        state.loc[:, "lag_1"] = predicted_return
        state.loc[:, "rolling_mean"] = np.mean(returns[-12:])
        state.loc[:, "rolling_volatility"] = np.std(returns[-12:]) if len(returns) > 1 else state["rolling_volatility"].iloc[0]
        state.loc[:, "momentum_6m"] = np.prod([1 + r for r in returns[-6:]]) - 1
    return _path_from_return_sequence(monthly, returns, metrics, "Random forest regression")


def monte_carlo_forecast(monthly: pd.DataFrame, horizon_months: int, simulations: int = 1000) -> dict:
    returns = monthly["monthly_return"].dropna()
    mu, sigma = returns.mean(), returns.std()
    last_price = monthly["adjusted_close"].dropna().iloc[-1]
    rng = np.random.default_rng(42)
    simulated_returns = rng.normal(mu, sigma, size=(simulations, horizon_months))
    paths = last_price * np.cumprod(1 + simulated_returns, axis=1)
    dates = _future_index(monthly["month_start"].iloc[-1], horizon_months)
    forecast = pd.DataFrame(
        {
            "date_day": dates,
            "forecast": np.percentile(paths, 50, axis=0),
            "lower": np.percentile(paths, 5, axis=0),
            "upper": np.percentile(paths, 95, axis=0),
            "scenario": "monte_carlo",
        }
    )
    final_returns = paths[:, -1] / last_price - 1
    drawdowns = paths / np.maximum.accumulate(paths, axis=1) - 1
    metrics = {
        "probability_positive_return": float((final_returns > 0).mean()),
        "probability_drawdown_gt_20pct": float((drawdowns.min(axis=1) < -0.2).mean()),
        "mae": np.nan,
        "rmse": np.nan,
        "mape": np.nan,
    }
    return {"forecast": forecast, "metrics": metrics, "model": "Monte Carlo simulation"}


def _path_from_return_sequence(monthly: pd.DataFrame, returns: list[float], metrics: dict, model: str) -> dict:
    last_price = monthly["adjusted_close"].dropna().iloc[-1]
    dates = _future_index(monthly["month_start"].iloc[-1], len(returns))
    forecast_price = last_price * np.cumprod([1 + r for r in returns])
    sigma = monthly["monthly_return"].dropna().std()
    steps = np.arange(1, len(returns) + 1)
    band = 1.96 * sigma * np.sqrt(steps)
    forecast = pd.DataFrame({"date_day": dates, "forecast": forecast_price})
    forecast["lower"] = forecast["forecast"] * (1 - band)
    forecast["upper"] = forecast["forecast"] * (1 + band)
    forecast["scenario"] = "base"
    return {"forecast": forecast, "metrics": metrics, "model": model}
