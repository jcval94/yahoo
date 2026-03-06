pd = __import__("pytest").importorskip("pandas")

from src import edge_drift


def test_tolerance_varies_by_regime():
    base = edge_drift._tolerance_for_regime("no_event_day")
    earnings = edge_drift._tolerance_for_regime("earnings_day")

    assert earnings["MAE"] > base["MAE"]
    assert earnings["MSE"] > base["MSE"]
    assert earnings["R2"] < base["R2"]


def test_evaluate_drift_includes_regime_column():
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
    assert "regime" in result.columns
    assert result.loc[0, "regime"] == "macro_day"
