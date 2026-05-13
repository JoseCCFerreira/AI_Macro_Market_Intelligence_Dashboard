from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SQLITE_PATH = DATA_DIR / "relational_market.db"
DUCKDB_PATH = DATA_DIR / "analytics_market.duckdb"
ASSET_CONFIG_PATH = PROJECT_ROOT / "config" / "assets.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dirs() -> None:
    for path in [DATA_DIR, DATA_DIR / "raw", DATA_DIR / "processed"]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_region(region: str) -> str:
    if region in {"ETFs/Futures", "Miners", "Commodities"}:
        return "Global"
    return region


def safe_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
