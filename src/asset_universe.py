from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .utils import ASSET_CONFIG_PATH, normalize_region, unique_preserve_order


@dataclass(frozen=True)
class AssetUniverse:
    assets: pd.DataFrame
    benchmarks: dict[str, list[str]]


def _asset_type(sector: str, bucket: str, ticker: str) -> str:
    if "=" in ticker:
        return "Future"
    if sector == "Precious Metals" and bucket in {"ETFs/Futures"}:
        return "ETF/Future"
    if sector == "Precious Metals" and bucket == "Miners":
        return "Equity"
    if ticker in {"SPY", "QQQ", "ACWI", "FEZ", "EWJ", "FXI", "EEM"}:
        return "ETF"
    return "Equity"


def read_asset_universe(path: str | Path = ASSET_CONFIG_PATH) -> AssetUniverse:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    validate_config(payload)
    rows: list[dict] = []
    for sector, by_region in payload["sectors"].items():
        for region_bucket, tickers in by_region.items():
            region = normalize_region(region_bucket)
            for ticker in tickers:
                rows.append(
                    {
                        "ticker": str(ticker).strip(),
                        "asset_name": str(ticker).strip(),
                        "sector": sector,
                        "region": region,
                        "asset_type": _asset_type(sector, region_bucket, str(ticker)),
                        "currency": None,
                        "exchange": None,
                        "is_active": True,
                    }
                )

    benchmark_rows = []
    for region, tickers in payload.get("benchmarks", {}).items():
        for ticker in tickers:
            benchmark_rows.append(
                {
                    "ticker": ticker,
                    "asset_name": ticker,
                    "sector": "Benchmark",
                    "region": region,
                    "asset_type": "ETF",
                    "currency": None,
                    "exchange": None,
                    "is_active": True,
                }
            )

    df = pd.DataFrame(rows + benchmark_rows).drop_duplicates("ticker").reset_index(drop=True)
    return AssetUniverse(assets=df, benchmarks=payload.get("benchmarks", {}))


def validate_config(payload: dict) -> None:
    if not isinstance(payload, dict) or "sectors" not in payload:
        raise ValueError("assets.yaml must contain a top-level 'sectors' mapping.")
    for sector, by_region in payload["sectors"].items():
        if not isinstance(by_region, dict):
            raise ValueError(f"Sector '{sector}' must map to regions.")
        for region, tickers in by_region.items():
            if not isinstance(tickers, list) or not tickers:
                raise ValueError(f"{sector}/{region} must contain a non-empty ticker list.")


def get_assets(
    sectors: list[str] | None = None,
    regions: list[str] | None = None,
    include_benchmarks: bool = True,
) -> pd.DataFrame:
    universe = read_asset_universe()
    df = universe.assets.copy()
    if not include_benchmarks:
        df = df[df["sector"] != "Benchmark"]
    if sectors and "All" not in sectors:
        df = df[df["sector"].isin(sectors)]
    if regions and "Global" not in regions:
        df = df[df["region"].isin(regions)]
    return df.reset_index(drop=True)


def all_tickers(include_benchmarks: bool = True) -> list[str]:
    df = get_assets(include_benchmarks=include_benchmarks)
    return unique_preserve_order(df["ticker"].tolist())
