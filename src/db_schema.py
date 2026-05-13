SQLITE_DDL = [
    "PRAGMA foreign_keys = ON",
    """
    CREATE TABLE IF NOT EXISTS dim_sector (
        sector_id INTEGER PRIMARY KEY,
        sector_name TEXT UNIQUE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dim_region (
        region_id INTEGER PRIMARY KEY,
        region_name TEXT UNIQUE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dim_asset (
        asset_id INTEGER PRIMARY KEY,
        ticker TEXT UNIQUE NOT NULL,
        asset_name TEXT,
        sector TEXT,
        region TEXT,
        asset_type TEXT,
        currency TEXT,
        exchange TEXT,
        is_active BOOLEAN,
        created_at TIMESTAMP,
        updated_at TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS download_batch (
        batch_id INTEGER PRIMARY KEY,
        download_timestamp TIMESTAMP,
        start_date DATE,
        end_date DATE,
        frequency TEXT,
        source TEXT,
        status TEXT,
        requested_tickers INTEGER,
        successful_tickers INTEGER,
        failed_tickers INTEGER,
        inserted_rows INTEGER,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_price_raw (
        price_id INTEGER PRIMARY KEY,
        batch_id INTEGER,
        asset_id INTEGER,
        date DATE,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adjusted_close REAL,
        volume REAL,
        dividends REAL,
        stock_splits REAL,
        source TEXT,
        loaded_at TIMESTAMP,
        FOREIGN KEY(batch_id) REFERENCES download_batch(batch_id),
        FOREIGN KEY(asset_id) REFERENCES dim_asset(asset_id),
        UNIQUE(batch_id, asset_id, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_quality_log (
        log_id INTEGER PRIMARY KEY,
        batch_id INTEGER,
        asset_id INTEGER,
        issue_type TEXT,
        issue_description TEXT,
        severity TEXT,
        created_at TIMESTAMP,
        FOREIGN KEY(batch_id) REFERENCES download_batch(batch_id),
        FOREIGN KEY(asset_id) REFERENCES dim_asset(asset_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS failed_ticker_log (
        failed_id INTEGER PRIMARY KEY,
        batch_id INTEGER,
        ticker TEXT,
        error_message TEXT,
        created_at TIMESTAMP,
        FOREIGN KEY(batch_id) REFERENCES download_batch(batch_id)
    )
    """,
]


DUCKDB_DDL = [
    """
    CREATE TABLE IF NOT EXISTS dim_asset (
        asset_id INTEGER,
        ticker VARCHAR,
        asset_name VARCHAR,
        sector VARCHAR,
        region VARCHAR,
        asset_type VARCHAR,
        currency VARCHAR,
        exchange VARCHAR,
        is_active BOOLEAN
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dim_date (
        date_key INTEGER,
        date_day DATE,
        year INTEGER,
        quarter INTEGER,
        month INTEGER,
        month_name VARCHAR,
        week INTEGER,
        day INTEGER,
        day_of_week VARCHAR,
        is_weekend BOOLEAN
    )
    """,
    "CREATE TABLE IF NOT EXISTS dim_sector (sector_id INTEGER, sector_name VARCHAR)",
    "CREATE TABLE IF NOT EXISTS dim_region (region_id INTEGER, region_name VARCHAR)",
    """
    CREATE TABLE IF NOT EXISTS fact_market_prices (
        asset_id INTEGER,
        date_key INTEGER,
        date_day DATE,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        adjusted_close DOUBLE,
        volume DOUBLE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fact_daily_returns (
        asset_id INTEGER,
        date_key INTEGER,
        date_day DATE,
        daily_return DOUBLE,
        log_return DOUBLE,
        cumulative_return DOUBLE,
        rolling_return_7d DOUBLE,
        rolling_return_30d DOUBLE,
        rolling_return_90d DOUBLE,
        rolling_volatility_30d DOUBLE,
        rolling_volatility_90d DOUBLE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fact_monthly_returns (
        asset_id INTEGER,
        month_key INTEGER,
        month_start DATE,
        monthly_return DOUBLE,
        cumulative_monthly_return DOUBLE,
        monthly_volatility DOUBLE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fact_asset_features (
        asset_id INTEGER,
        calculation_date DATE,
        return_1m DOUBLE,
        return_3m DOUBLE,
        return_6m DOUBLE,
        return_1y DOUBLE,
        return_3y_annualized DOUBLE,
        annualized_return DOUBLE,
        annualized_volatility DOUBLE,
        sharpe_ratio DOUBLE,
        sortino_ratio DOUBLE,
        max_drawdown DOUBLE,
        skewness DOUBLE,
        kurtosis DOUBLE,
        var_95 DOUBLE,
        cvar_95 DOUBLE,
        beta_vs_benchmark DOUBLE,
        correlation_vs_benchmark DOUBLE,
        average_volume DOUBLE,
        volume_volatility DOUBLE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fact_detected_events (
        event_id INTEGER,
        detected_at TIMESTAMP,
        asset_id INTEGER,
        date_day DATE,
        event_type VARCHAR,
        event_value DOUBLE,
        price DOUBLE,
        volume DOUBLE,
        severity VARCHAR,
        explanation VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mart_asset_performance AS
    SELECT * FROM (SELECT NULL::INTEGER asset_id, NULL::VARCHAR ticker, NULL::VARCHAR sector,
    NULL::VARCHAR region, NULL::DOUBLE latest_price, NULL::DOUBLE ytd_return,
    NULL::DOUBLE return_1y, NULL::DOUBLE return_3y, NULL::DOUBLE return_5y,
    NULL::DOUBLE annualized_return, NULL::DOUBLE annualized_volatility,
    NULL::DOUBLE sharpe_ratio, NULL::DOUBLE max_drawdown) WHERE FALSE
    """,
    """
    CREATE TABLE IF NOT EXISTS mart_regional_performance AS
    SELECT * FROM (SELECT NULL::VARCHAR region, NULL::DATE date_day, NULL::DOUBLE equal_weighted_return,
    NULL::DOUBLE cumulative_return, NULL::DOUBLE volatility, NULL::DOUBLE sharpe_ratio,
    NULL::DOUBLE max_drawdown) WHERE FALSE
    """,
    """
    CREATE TABLE IF NOT EXISTS mart_sector_performance AS
    SELECT * FROM (SELECT NULL::VARCHAR sector, NULL::DATE date_day, NULL::DOUBLE equal_weighted_return,
    NULL::DOUBLE cumulative_return, NULL::DOUBLE volatility, NULL::DOUBLE sharpe_ratio,
    NULL::DOUBLE max_drawdown) WHERE FALSE
    """,
    """
    CREATE TABLE IF NOT EXISTS mart_forecasting_input AS
    SELECT * FROM (SELECT NULL::INTEGER asset_id, NULL::DATE date_day, NULL::DOUBLE adjusted_close,
    NULL::DOUBLE monthly_return, NULL::DOUBLE rolling_mean, NULL::DOUBLE rolling_volatility,
    NULL::DOUBLE momentum_3m, NULL::DOUBLE momentum_6m, NULL::DOUBLE momentum_12m) WHERE FALSE
    """,
    """
    CREATE TABLE IF NOT EXISTS mart_clustering_input AS
    SELECT * FROM (SELECT NULL::INTEGER asset_id, NULL::VARCHAR ticker, NULL::VARCHAR sector,
    NULL::VARCHAR region, NULL::DOUBLE return_1m, NULL::DOUBLE return_3m, NULL::DOUBLE return_6m,
    NULL::DOUBLE return_1y, NULL::DOUBLE return_3y_annualized, NULL::DOUBLE annualized_volatility,
    NULL::DOUBLE sharpe_ratio, NULL::DOUBLE max_drawdown, NULL::DOUBLE skewness, NULL::DOUBLE kurtosis,
    NULL::DOUBLE average_volume, NULL::DOUBLE beta_vs_benchmark, NULL::DOUBLE correlation_vs_benchmark)
    WHERE FALSE
    """,
]
