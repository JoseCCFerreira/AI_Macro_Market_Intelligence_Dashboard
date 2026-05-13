from src.etl_pipeline import PipelineStatus


def test_pipeline_status_output():
    status = PipelineStatus(
        status="success",
        started_at="2026-01-01T00:00:00Z",
        batch_id=1,
        requested_tickers=2,
        successful_tickers=2,
        failed_tickers=0,
        rows_inserted_sqlite=10,
        rows_loaded_duckdb=20,
        data_quality_warnings=0,
        failed_symbols=[],
        message="ok",
    )
    payload = status.to_dict()
    assert payload["status"] == "success"
    assert payload["rows_loaded_duckdb"] == 20
