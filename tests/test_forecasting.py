import pandas as pd

from src.forecasting import forecast_asset, prepare_monthly_forecasting_dataset


def test_forecasting_input_and_baseline():
    dates = pd.date_range("2020-01-01", periods=900, freq="D")
    prices = pd.DataFrame({"date_day": dates, "adjusted_close": range(100, 1000)})
    monthly = prepare_monthly_forecasting_dataset(prices)
    result = forecast_asset(monthly, horizon_years=1, model_name="Historical average")
    assert len(result["forecast"]) == 12
    assert {"forecast", "lower", "upper"}.issubset(result["forecast"].columns)
