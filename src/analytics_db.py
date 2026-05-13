from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .db_schema import DUCKDB_DDL
from .utils import DUCKDB_PATH, ensure_dirs


def connect(path: str | Path = DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    ensure_dirs()
    return duckdb.connect(str(path))


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    for statement in DUCKDB_DDL:
        conn.execute(statement)


def replace_table(conn: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    if df.empty:
        create_schema(conn)
        return
    conn.register("_df_to_load", df)
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _df_to_load")
    conn.unregister("_df_to_load")


def query_df(sql: str, params: tuple | None = None, path: str | Path = DUCKDB_PATH) -> pd.DataFrame:
    with connect(path) as conn:
        return conn.execute(sql, params or ()).fetchdf()


def load_analytical_tables(
    conn: duckdb.DuckDBPyConnection,
    tables: dict[str, pd.DataFrame],
) -> None:
    create_schema(conn)
    for name, df in tables.items():
        replace_table(conn, name, df)


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        (table_name,),
    ).fetchone()[0]
    return result > 0
