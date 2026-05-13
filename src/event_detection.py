from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import utc_now


def _severity(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 0.15:
        return "extreme"
    if magnitude >= 0.08:
        return "high"
    if magnitude >= 0.04:
        return "medium"
    return "low"


def detect_events(clean: pd.DataFrame, daily_returns: pd.DataFrame, gain_threshold: float = 0.05, loss_threshold: float = -0.05) -> pd.DataFrame:
    events: list[dict] = []
    if clean.empty or daily_returns.empty:
        return pd.DataFrame()
    enriched = daily_returns.merge(
        clean[["asset_id", "date", "ticker", "asset_name", "sector", "region", "adjusted_close", "volume"]],
        left_on=["asset_id", "date_day"],
        right_on=["asset_id", "date"],
        how="left",
    )
    for asset_id, group in enriched.groupby("asset_id"):
        g = group.sort_values("date_day").copy()
        g["weekly_return"] = g["adjusted_close"].pct_change(5)
        g["rolling_52w_high"] = g["adjusted_close"].rolling(252, min_periods=30).max()
        g["rolling_52w_low"] = g["adjusted_close"].rolling(252, min_periods=30).min()
        g["volume_z"] = (g["volume"] - g["volume"].rolling(60).mean()) / g["volume"].rolling(60).std()
        g["vol_z"] = (g["daily_return"].rolling(20).std() - g["daily_return"].rolling(120).std()) / g["daily_return"].rolling(120).std()
        g["drawdown"] = g["adjusted_close"] / g["adjusted_close"].cummax() - 1
        for row in g.itertuples(index=False):
            checks = [
                ("Daily big gain", row.daily_return, pd.notna(row.daily_return) and row.daily_return >= gain_threshold),
                ("Daily big loss", row.daily_return, pd.notna(row.daily_return) and row.daily_return <= loss_threshold),
                ("Weekly big gain", row.weekly_return, pd.notna(row.weekly_return) and row.weekly_return >= gain_threshold * 1.5),
                ("Weekly big loss", row.weekly_return, pd.notna(row.weekly_return) and row.weekly_return <= loss_threshold * 1.5),
                ("New 52-week high", 0.0, pd.notna(row.rolling_52w_high) and row.adjusted_close >= row.rolling_52w_high),
                ("New 52-week low", 0.0, pd.notna(row.rolling_52w_low) and row.adjusted_close <= row.rolling_52w_low),
                ("Large volume spike", row.volume_z, pd.notna(row.volume_z) and row.volume_z >= 3),
                ("Volatility spike", row.vol_z, pd.notna(row.vol_z) and row.vol_z >= 1.5),
                ("Drawdown alert", row.drawdown, pd.notna(row.drawdown) and row.drawdown <= -0.2),
                ("Strong rebound", row.daily_return, pd.notna(row.daily_return) and row.daily_return >= 0.04 and pd.notna(row.drawdown) and row.drawdown < -0.1),
            ]
            for event_type, value, ok in checks:
                if ok:
                    value = float(value) if np.isfinite(value) else 0.0
                    events.append(
                        {
                            "detected_at": utc_now(),
                            "asset_id": int(asset_id),
                            "date_day": pd.Timestamp(row.date_day).date(),
                            "event_type": event_type,
                            "event_value": value,
                            "price": float(row.adjusted_close),
                            "volume": float(row.volume or 0),
                            "severity": _severity(value if value else row.daily_return if pd.notna(row.daily_return) else 0),
                            "explanation": f"{row.ticker} triggered {event_type} on {pd.Timestamp(row.date_day).date()}",
                        }
                    )
    out = pd.DataFrame(events)
    if not out.empty:
        out.insert(0, "event_id", range(1, len(out) + 1))
    return out
