from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .db_schema import SQLITE_DDL
from .utils import SQLITE_PATH, ensure_dirs, utc_now


def connect(path: str | Path = SQLITE_PATH) -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    for statement in SQLITE_DDL:
        conn.execute(statement)
    conn.commit()


def upsert_reference_data(conn: sqlite3.Connection, assets: pd.DataFrame) -> pd.DataFrame:
    now = utc_now()
    for sector in sorted(assets["sector"].dropna().unique()):
        conn.execute("INSERT OR IGNORE INTO dim_sector(sector_name) VALUES (?)", (sector,))
    for region in sorted(assets["region"].dropna().unique()):
        conn.execute("INSERT OR IGNORE INTO dim_region(region_name) VALUES (?)", (region,))
    for row in assets.to_dict("records"):
        conn.execute(
            """
            INSERT INTO dim_asset(ticker, asset_name, sector, region, asset_type, currency, exchange,
                                  is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                asset_name=excluded.asset_name,
                sector=excluded.sector,
                region=excluded.region,
                asset_type=excluded.asset_type,
                currency=COALESCE(excluded.currency, dim_asset.currency),
                exchange=COALESCE(excluded.exchange, dim_asset.exchange),
                is_active=excluded.is_active,
                updated_at=excluded.updated_at
            """,
            (
                row["ticker"],
                row.get("asset_name"),
                row.get("sector"),
                row.get("region"),
                row.get("asset_type"),
                row.get("currency"),
                row.get("exchange"),
                bool(row.get("is_active", True)),
                now,
                now,
            ),
        )
    conn.commit()
    return pd.read_sql_query("SELECT * FROM dim_asset", conn)


def create_download_batch(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    frequency: str,
    requested_tickers: int,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO download_batch(download_timestamp, start_date, end_date, frequency, source, status,
                                   requested_tickers, successful_tickers, failed_tickers, inserted_rows, notes)
        VALUES (?, ?, ?, ?, 'yfinance', 'running', ?, 0, 0, 0, '')
        """,
        (utc_now(), start_date, end_date, frequency, requested_tickers),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_download_batch(conn: sqlite3.Connection, batch_id: int, status: dict) -> None:
    conn.execute(
        """
        UPDATE download_batch
        SET status=?, successful_tickers=?, failed_tickers=?, inserted_rows=?, notes=?
        WHERE batch_id=?
        """,
        (
            status.get("status", "success"),
            int(status.get("successful_tickers", 0)),
            int(status.get("failed_tickers", 0)),
            int(status.get("inserted_rows", 0)),
            status.get("notes", ""),
            batch_id,
        ),
    )
    conn.commit()


def insert_raw_prices(conn: sqlite3.Connection, batch_id: int, prices: pd.DataFrame) -> int:
    if prices.empty:
        return 0
    assets = pd.read_sql_query("SELECT asset_id, ticker FROM dim_asset", conn)
    df = prices.merge(assets, on="ticker", how="inner")
    now = utc_now()
    rows = [
        (
            batch_id,
            int(row.asset_id),
            str(pd.Timestamp(row.date).date()),
            row.open,
            row.high,
            row.low,
            row.close,
            row.adjusted_close,
            row.volume,
            getattr(row, "dividends", 0.0),
            getattr(row, "stock_splits", 0.0),
            "yfinance",
            now,
        )
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT OR IGNORE INTO market_price_raw(batch_id, asset_id, date, open, high, low, close,
            adjusted_close, volume, dividends, stock_splits, source, loaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def insert_failed_tickers(conn: sqlite3.Connection, batch_id: int, failures: list[dict]) -> None:
    if not failures:
        return
    conn.executemany(
        "INSERT INTO failed_ticker_log(batch_id, ticker, error_message, created_at) VALUES (?, ?, ?, ?)",
        [(batch_id, item["ticker"], item["error"], utc_now()) for item in failures],
    )
    conn.commit()


def insert_quality_logs(conn: sqlite3.Connection, batch_id: int, issues: pd.DataFrame) -> None:
    if issues.empty:
        return
    assets = pd.read_sql_query("SELECT asset_id, ticker FROM dim_asset", conn)
    df = issues.merge(assets, on="ticker", how="left")
    rows = [
        (
            batch_id,
            None if pd.isna(row.asset_id) else int(row.asset_id),
            row.issue_type,
            row.issue_description,
            row.severity,
            utc_now(),
        )
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        """
        INSERT INTO data_quality_log(batch_id, asset_id, issue_type, issue_description, severity, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def read_raw_for_transformation(conn: sqlite3.Connection, batch_id: int | None = None) -> pd.DataFrame:
    where = "" if batch_id is None else "WHERE p.batch_id = ?"
    params = () if batch_id is None else (batch_id,)
    return pd.read_sql_query(
        f"""
        SELECT p.*, a.ticker, a.asset_name, a.sector, a.region, a.asset_type, a.currency, a.exchange, a.is_active
        FROM market_price_raw p
        JOIN dim_asset a ON p.asset_id = a.asset_id
        {where}
        ORDER BY a.ticker, p.date
        """,
        conn,
        params=params,
        parse_dates=["date"],
    )


def read_quality_logs(conn: sqlite3.Connection, limit: int = 200) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT q.created_at, a.ticker, q.issue_type, q.issue_description, q.severity
        FROM data_quality_log q
        LEFT JOIN dim_asset a ON q.asset_id = a.asset_id
        ORDER BY q.log_id DESC
        LIMIT ?
        """,
        conn,
        params=(limit,),
    )
