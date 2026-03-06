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


def test_enrich_regime_labels_adds_requested_regimes():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "model": ["rf", "rf"],
            "pred": [101.0, 99.0],
            "real": [100.0, 98.0],
            "real_inc": [0.02, -0.03],
            "timestamp": ["2025-01-01 10:00:00", "2025-01-01 16:30:00"],
        }
    )

    out = evaluation.enrich_regime_labels(df)

    assert set(["high_vol_regime", "risk_regime", "session_bucket"]).issubset(out.columns)
    assert set(out["risk_regime"]) == {"risk_on", "risk_off"}


def test_evaluate_regime_metrics_compares_with_baseline():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "model": ["rf", "rf"],
            "pred": [101.0, 103.0],
            "baseline_pred": [99.0, 99.0],
            "real": [100.0, 100.0],
            "real_inc": [0.01, 0.03],
            "timestamp": ["2025-01-01 10:00:00", "2025-01-01 12:00:00"],
        }
    )

    regime_df = evaluation.evaluate_regime_metrics(df)

    assert not regime_df.empty
    assert "baseline_MAE" in regime_df.columns
    assert "delta_vs_baseline_MAE" in regime_df.columns


def test_save_regime_reports_saves_per_ticker_model_regime_type(tmp_path):
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "model": ["rf", "rf", "xgb"],
            "pred": [101.0, 103.0, 100.0],
            "real": [100.0, 100.0, 100.0],
            "real_inc": [0.01, 0.03, -0.01],
            "timestamp": ["2025-01-01 10:00:00", "2025-01-01 12:00:00", "2025-01-01 16:30:00"],
        }
    )

    files = evaluation.save_regime_reports(df, tmp_path)

    assert files
    assert all(path.exists() for path in files)
