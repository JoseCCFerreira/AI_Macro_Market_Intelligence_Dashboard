from __future__ import annotations

import pandas as pd


def clean_market_data(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["asset_id", "ticker", "date", "adjusted_close"])
    price_cols = ["open", "high", "low", "close", "adjusted_close"]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    df = df[(df[price_cols] > 0).all(axis=1)]
    df = df.sort_values(["asset_id", "date"]).drop_duplicates(["asset_id", "date"], keep="last")
    return df.reset_index(drop=True)


def resample_monthly(clean: pd.DataFrame) -> pd.DataFrame:
    if clean.empty:
        return pd.DataFrame()
    frames = []
    for asset_id, group in clean.groupby("asset_id"):
        g = group.set_index("date").sort_index()
        monthly = g.resample("MS").agg({"adjusted_close": "last", "volume": "sum"}).dropna()
        monthly["asset_id"] = asset_id
        monthly["monthly_return"] = monthly["adjusted_close"].pct_change()
        frames.append(monthly.reset_index().rename(columns={"date": "month_start"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
