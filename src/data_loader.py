from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass
class DownloadResult:
    prices: pd.DataFrame
    failures: list[dict]


def _standardize_history(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.reset_index()
    if "Date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "Date"})
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    rename = {"adj_close": "adjusted_close", "stock_splits": "stock_splits"}
    out = out.rename(columns=rename)
    if "adjusted_close" not in out.columns and "close" in out.columns:
        out["adjusted_close"] = out["close"]
    for col in ["open", "high", "low", "close", "adjusted_close", "volume", "dividends", "stock_splits"]:
        if col not in out.columns:
            out[col] = 0.0
    out["ticker"] = ticker
    return out[
        ["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume", "dividends", "stock_splits"]
    ]


def download_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    frequency: str = "1d",
    retries: int = 2,
    pause_seconds: float = 0.5,
) -> DownloadResult:
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    for ticker in tickers:
        last_error = ""
        for attempt in range(retries + 1):
            try:
                data = yf.Ticker(ticker).history(
                    start=start_date,
                    end=end_date,
                    interval=frequency,
                    auto_adjust=False,
                    actions=True,
                )
                standardized = _standardize_history(ticker, data)
                if standardized.empty:
                    raise ValueError("No data returned")
                frames.append(standardized)
                break
            except Exception as exc:  # yfinance errors vary by endpoint and symbol
                last_error = str(exc)
                if attempt < retries:
                    time.sleep(pause_seconds)
        else:
            failures.append({"ticker": ticker, "error": last_error or "Unknown download error"})
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return DownloadResult(prices=prices, failures=failures)
