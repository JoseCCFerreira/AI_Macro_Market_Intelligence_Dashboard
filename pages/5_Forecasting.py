from __future__ import annotations

import streamlit as st

from src.app_support import DISCLAIMER, apply_theme, empty_state, filters, load_table, run_query
from src.forecasting import forecast_asset, prepare_monthly_forecasting_dataset
from src.plotting import forecast_chart
from src.storage import dataframe_csv_download


apply_theme()
st.title("Forecasting")
selected = filters()
st.markdown(f"<p class='risk-note'>{DISCLAIMER} Forecasts are scenario bands, not guarantees.</p>", unsafe_allow_html=True)

assets = load_table("dim_asset")
if assets.empty:
    empty_state()
    st.stop()

ticker = st.selectbox("Asset", assets["ticker"].sort_values().tolist())
history = run_query(
    f"""
    SELECT p.date_day, p.adjusted_close, a.ticker
    FROM fact_market_prices p JOIN dim_asset a USING(asset_id)
    WHERE a.ticker = '{ticker}'
    ORDER BY p.date_day
    """
)
monthly = prepare_monthly_forecasting_dataset(history)
result = forecast_asset(monthly, horizon_years=selected["horizon"], model_name=selected["model"])
forecast = result["forecast"]

st.plotly_chart(forecast_chart(history, forecast, ticker), use_container_width=True)
col1, col2, col3 = st.columns(3)
metrics = result.get("metrics", {})
col1.metric("Model", result.get("model", selected["model"]))
col2.metric("MAE", f"{metrics.get('mae', float('nan')):.4f}" if isinstance(metrics.get("mae"), float) else "n/a")
col3.metric("RMSE", f"{metrics.get('rmse', float('nan')):.4f}" if isinstance(metrics.get("rmse"), float) else "n/a")

if "probability_positive_return" in metrics:
    st.info(
        f"Monte Carlo scenario: probability of positive return {metrics['probability_positive_return']:.1%}; "
        f"probability of drawdown greater than 20% {metrics['probability_drawdown_gt_20pct']:.1%}."
    )
if "error" in metrics:
    st.warning(metrics["error"])

st.dataframe(forecast, use_container_width=True, hide_index=True)
st.download_button("Export forecast CSV", dataframe_csv_download(forecast), f"{ticker}_forecast.csv", "text/csv")
