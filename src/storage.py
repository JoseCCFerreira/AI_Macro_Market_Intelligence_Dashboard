from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import DATA_DIR, to_csv_bytes


def export_dataframe(df: pd.DataFrame, filename: str) -> Path:
    export_dir = DATA_DIR / "processed"
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / filename
    df.to_csv(path, index=False)
    return path


def dataframe_csv_download(df: pd.DataFrame) -> bytes:
    return to_csv_bytes(df)
