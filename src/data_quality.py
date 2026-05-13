from __future__ import annotations

import numpy as np
import pandas as pd


def validate_market_data(prices: pd.DataFrame) -> pd.DataFrame:
    issues: list[dict] = []
    if prices.empty:
        return pd.DataFrame(
            [{"ticker": "ALL", "issue_type": "empty_dataset", "issue_description": "No prices available", "severity": "high"}]
        )
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for ticker, group in df.groupby("ticker", dropna=False):
        prefix = {"ticker": ticker}
        if group["date"].isna().any():
            issues.append({**prefix, "issue_type": "invalid_dates", "issue_description": "Invalid dates detected", "severity": "high"})
        duplicate_count = group.duplicated(["ticker", "date"]).sum()
        if duplicate_count:
            issues.append({**prefix, "issue_type": "duplicate_records", "issue_description": f"{duplicate_count} duplicate ticker-date rows", "severity": "medium"})
        for col in ["open", "high", "low", "close", "adjusted_close"]:
            missing = group[col].isna().sum()
            non_positive = (group[col].fillna(1) <= 0).sum()
            if missing:
                issues.append({**prefix, "issue_type": f"missing_{col}", "issue_description": f"{missing} missing {col} values", "severity": "medium"})
            if non_positive:
                issues.append({**prefix, "issue_type": f"non_positive_{col}", "issue_description": f"{non_positive} zero/negative {col} values", "severity": "high"})
        if "volume" in group and group["volume"].isna().any():
            issues.append({**prefix, "issue_type": "missing_volume", "issue_description": "Missing volume values", "severity": "low"})
        returns = group.sort_values("date")["adjusted_close"].pct_change()
        extreme = returns[np.isfinite(returns) & (returns.abs() > 0.25)]
        if not extreme.empty:
            issues.append({**prefix, "issue_type": "extreme_returns", "issue_description": f"{len(extreme)} daily moves above 25%", "severity": "medium"})
        if len(group) < 252:
            issues.append({**prefix, "issue_type": "short_history", "issue_description": "Less than one trading year of history", "severity": "medium"})
    return pd.DataFrame(issues, columns=["ticker", "issue_type", "issue_description", "severity"])


def has_blocking_quality_issue(issues: pd.DataFrame) -> bool:
    if issues.empty:
        return False
    return bool((issues["severity"] == "high").any())
