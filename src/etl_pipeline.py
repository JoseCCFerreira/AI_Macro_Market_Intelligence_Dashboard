from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from . import analytics_db, relational_db
from .asset_universe import read_asset_universe
from .cleanup import delete_temporary_files, remove_partial_refresh
from .data_loader import download_market_data
from .data_quality import validate_market_data
from .event_detection import detect_events
from .feature_engineering import (
    build_asset_features,
    build_daily_returns,
    build_forecasting_input,
    build_group_performance,
    build_monthly_returns,
    date_dimension,
)
from .preprocessing import clean_market_data
from .utils import ASSET_CONFIG_PATH, DUCKDB_PATH, SQLITE_PATH, ensure_dirs, utc_now


@dataclass
class PipelineStatus:
    status: str
    started_at: str
    batch_id: int | None
    requested_tickers: int
    successful_tickers: int
    failed_tickers: int
    rows_inserted_sqlite: int
    rows_loaded_duckdb: int
    data_quality_warnings: int
    failed_symbols: list[str]
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_full_refresh(
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    frequency: str = "1d",
    selected_tickers: list[str] | None = None,
    benchmark_ticker: str = "SPY",
) -> PipelineStatus:
    ensure_dirs()
    started_at = utc_now()
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    batch_id: int | None = None
    try:
        universe = read_asset_universe(ASSET_CONFIG_PATH)
        assets = universe.assets
        if selected_tickers:
            assets = assets[assets["ticker"].isin(selected_tickers)]
        tickers = assets["ticker"].drop_duplicates().tolist()
        with relational_db.connect(SQLITE_PATH) as sqlite_conn:
            relational_db.create_schema(sqlite_conn)
            relational_db.upsert_reference_data(sqlite_conn, universe.assets)
            batch_id = relational_db.create_download_batch(sqlite_conn, start_date, end_date, frequency, len(tickers))
            result = download_market_data(tickers, start_date, end_date, frequency)
            inserted = relational_db.insert_raw_prices(sqlite_conn, batch_id, result.prices)
            relational_db.insert_failed_tickers(sqlite_conn, batch_id, result.failures)
            issues = validate_market_data(result.prices)
            relational_db.insert_quality_logs(sqlite_conn, batch_id, issues)
            raw = relational_db.read_raw_for_transformation(sqlite_conn)

        clean = clean_market_data(raw)
        if clean.empty:
            raise ValueError("No clean market records available after validation.")
        rows_loaded = _build_duckdb_atomic(clean, benchmark_ticker)
        delete_temporary_files()
        status_name = "partial_success" if result.failures or not issues.empty else "success"
        with relational_db.connect(SQLITE_PATH) as sqlite_conn:
            relational_db.update_download_batch(
                sqlite_conn,
                batch_id,
                {
                    "status": status_name,
                    "successful_tickers": len(tickers) - len(result.failures),
                    "failed_tickers": len(result.failures),
                    "inserted_rows": inserted,
                    "notes": f"{len(issues)} data quality warnings",
                },
            )
        return PipelineStatus(
            status=status_name,
            started_at=started_at,
            batch_id=batch_id,
            requested_tickers=len(tickers),
            successful_tickers=len(tickers) - len(result.failures),
            failed_tickers=len(result.failures),
            rows_inserted_sqlite=inserted,
            rows_loaded_duckdb=rows_loaded,
            data_quality_warnings=len(issues),
            failed_symbols=[item["ticker"] for item in result.failures],
            message="Refresh completed. Streamlit should consume the DuckDB analytical layer.",
        )
    except Exception as exc:
        if batch_id is not None:
            with relational_db.connect(SQLITE_PATH) as sqlite_conn:
                relational_db.update_download_batch(
                    sqlite_conn,
                    batch_id,
                    {"status": "failed", "notes": str(exc), "successful_tickers": 0, "failed_tickers": 0, "inserted_rows": 0},
                )
        return PipelineStatus(
            status="failed",
            started_at=started_at,
            batch_id=batch_id,
            requested_tickers=0,
            successful_tickers=0,
            failed_tickers=0,
            rows_inserted_sqlite=0,
            rows_loaded_duckdb=0,
            data_quality_warnings=0,
            failed_symbols=[],
            message=str(exc),
        )


def _build_duckdb_atomic(clean: pd.DataFrame, benchmark_ticker: str) -> int:
    tmp_path = Path(str(DUCKDB_PATH) + ".tmp")
    remove_partial_refresh(tmp_path)
    assets = clean[
        ["asset_id", "ticker", "asset_name", "sector", "region", "asset_type", "currency", "exchange", "is_active"]
    ].drop_duplicates("asset_id")
    dates = date_dimension(clean["date"])
    prices = clean[["asset_id", "date", "open", "high", "low", "close", "adjusted_close", "volume"]].copy()
    prices["date_key"] = pd.to_datetime(prices["date"]).dt.strftime("%Y%m%d").astype(int)
    prices = prices.rename(columns={"date": "date_day"})
    daily = build_daily_returns(clean)
    monthly = build_monthly_returns(clean)
    features = build_asset_features(clean, benchmark_ticker=benchmark_ticker)
    events = detect_events(clean, daily)
    regional = build_group_performance(daily, assets, "region").rename(columns={"region": "region"})
    sector = build_group_performance(daily, assets, "sector").rename(columns={"sector": "sector"})
    forecast_input = build_forecasting_input(clean)
    clustering_input = features.merge(assets[["asset_id", "ticker", "sector", "region"]], on="asset_id", how="left")
    performance = _asset_performance_mart(clean, features, assets)
    sectors = assets[["sector"]].drop_duplicates().sort_values("sector").reset_index(drop=True)
    sectors.insert(0, "sector_id", range(1, len(sectors) + 1))
    sectors = sectors.rename(columns={"sector": "sector_name"})
    regions = assets[["region"]].drop_duplicates().sort_values("region").reset_index(drop=True)
    regions.insert(0, "region_id", range(1, len(regions) + 1))
    regions = regions.rename(columns={"region": "region_name"})
    tables = {
        "dim_asset": assets,
        "dim_date": dates,
        "dim_sector": sectors,
        "dim_region": regions,
        "fact_market_prices": prices,
        "fact_daily_returns": daily,
        "fact_monthly_returns": monthly,
        "fact_asset_features": features,
        "fact_detected_events": events,
        "mart_asset_performance": performance,
        "mart_regional_performance": regional,
        "mart_sector_performance": sector,
        "mart_forecasting_input": forecast_input,
        "mart_clustering_input": clustering_input,
    }
    try:
        with analytics_db.connect(tmp_path) as conn:
            analytics_db.load_analytical_tables(conn, tables)
        tmp_path.replace(DUCKDB_PATH)
        return sum(len(df) for df in tables.values())
    except Exception:
        remove_partial_refresh(tmp_path)
        raise


def _asset_performance_mart(clean: pd.DataFrame, features: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset_id, group in clean.groupby("asset_id"):
        g = group.sort_values("date")
        latest = g.iloc[-1]
        start_year = g[g["date"].dt.year == latest["date"].year]
        ytd = latest["adjusted_close"] / start_year.iloc[0]["adjusted_close"] - 1 if not start_year.empty else None
        rows.append(
            {
                "asset_id": asset_id,
                "latest_price": latest["adjusted_close"],
                "ytd_return": ytd,
                "return_1y": _trailing_return(g, 252),
                "return_3y": _trailing_return(g, 252 * 3),
                "return_5y": _trailing_return(g, 252 * 5),
            }
        )
    perf = pd.DataFrame(rows)
    perf = perf.merge(assets[["asset_id", "ticker", "sector", "region"]], on="asset_id", how="left")
    perf = perf.merge(
        features[["asset_id", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"]],
        on="asset_id",
        how="left",
    )
    return perf[
        [
            "asset_id",
            "ticker",
            "sector",
            "region",
            "latest_price",
            "ytd_return",
            "return_1y",
            "return_3y",
            "return_5y",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ]
    ]


def _trailing_return(g: pd.DataFrame, days: int) -> float | None:
    if len(g) <= days:
        return None
    return float(g["adjusted_close"].iloc[-1] / g["adjusted_close"].iloc[-days] - 1)
