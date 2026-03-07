import pytest

pd = pytest.importorskip("pandas")

import src.training as training


class _DummyModel:
    def predict(self, X):
        return pd.Series(0.0, index=X.index)


def _build_df(start: str, periods: int = 200) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D")
    return pd.DataFrame(
        {
            "Close": pd.Series(range(periods), index=idx, dtype=float),
            "feat1": pd.Series(range(periods), index=idx, dtype=float),
        },
        index=idx,
    )


def test_train_models_does_not_reuse_previous_ticker_model(monkeypatch):
    calls = {"linear": 0, "perm": []}

    def fake_train_linear(X, y, cv=None):
        calls["linear"] += 1
        if calls["linear"] == 1:
            return _DummyModel()
        raise RuntimeError("forced linear training failure on second ticker")

    monkeypatch.setattr(training, "train_linear", fake_train_linear)
    monkeypatch.setattr(training, "train_rf", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rf disabled")))
    monkeypatch.setattr(training, "train_xgb", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("xgb disabled")))
    monkeypatch.setattr(training, "train_lgbm", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("lgbm disabled")))
    monkeypatch.setattr(training, "train_arima", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("arima disabled")))
    monkeypatch.setattr(training, "select_features_rf_cv", lambda *args, **kwargs: ["feat1"])
    monkeypatch.setattr(training, "hash_schema", lambda df: "hash")
    monkeypatch.setattr(training, "save_with_schema", lambda *args, **kwargs: None)
    monkeypatch.setattr(training, "evaluate_predictions", lambda y_true, y_pred: {"mae": 0.0})
    monkeypatch.setattr(training, "to_price", lambda values, base, mode: values)

    def fake_perm(*args, **kwargs):
        calls["perm"].append(kwargs["ticker"])

    monkeypatch.setattr(training, "_retrain_with_perm_importance", fake_perm)

    data = {
        "AAA": _build_df("2024-01-01"),
        "BBB": _build_df("2024-03-01"),
    }

    training.train_models(data)

    assert calls["linear"] == 2
    assert calls["perm"] == ["AAA"]
