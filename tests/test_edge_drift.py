pd = __import__("pytest").importorskip("pandas")

from src import edge_drift


def test_tolerance_varies_by_regime():
    base = edge_drift._tolerance_for_regime("no_event_day")
    earnings = edge_drift._tolerance_for_regime("earnings_day")

    assert earnings["MAE"] > base["MAE"]
    assert earnings["MSE"] > base["MSE"]
    assert earnings["R2"] < base["R2"]


def test_evaluate_drift_includes_regime_columns():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "model": ["rf", "rf"],
            "pred": [100.0, 110.0],
            "real": [100.0, 100.0],
            "event_type": ["macro_day", "macro_day"],
        }
    )

    result = edge_drift.evaluate_drift(df)

    assert not result.empty
    assert {"regime", "regime_type", "regime_only_failure"}.issubset(result.columns)
    assert "session_bucket" in set(result["regime_type"])


def test_evaluate_drift_flags_regime_only_failures():
    df = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "model": ["rf"] * 4,
            "pred": [101.0, 90.0, 100.0, 100.0],
            "real": [100.0, 100.0, 100.0, 100.0],
            "timestamp": [
                "2025-01-01 10:00:00",
                "2025-01-01 10:05:00",
                "2025-01-01 16:30:00",
                "2025-01-01 16:35:00",
            ],
            "real_inc": [0.01, 0.02, 0.0, 0.0],
        }
    )

    result = edge_drift.evaluate_drift(df)

    assert result["regime_only_failure"].any()
