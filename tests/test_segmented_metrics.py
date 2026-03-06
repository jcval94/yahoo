pd = __import__("pytest").importorskip("pandas")

from src import evaluation


def test_save_segmented_reports_creates_files(tmp_path):
    metrics_file = tmp_path / "metrics" / "edge_metrics_2025-01-01.csv"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "model": ["rf", "rf", "xgb"],
            "pred": [100.0, 102.0, 101.0],
            "real": [101.0, 103.0, 100.0],
            "timestamp": [
                "2025-01-01 10:00:00",
                "2025-01-01 12:00:00",
                "2025-01-01 16:30:00",
            ],
            "is_earnings_day": [True, False, False],
            "is_macro_day": [False, True, False],
        }
    )

    by_session, by_event = evaluation.save_segmented_reports(df, metrics_file)

    assert by_session.exists()
    assert by_event.exists()

    session_df = pd.read_csv(by_session)
    event_df = pd.read_csv(by_event)

    assert set(session_df["session"]) == {"opening", "midday", "after_hours"}
    assert set(event_df["event_type"]) == {"earnings_day", "macro_day", "no_event_day"}


def test_session_mapping_close_bucket():
    ts = pd.Timestamp("2025-01-01 15:15:00")
    assert evaluation._session_from_timestamp(ts) == "close"
