import importlib
import pandas as pd

from src import predict


def test_load_prediction_data_reuses_existing_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "CONFIG", {"etfs": ["AAA"], "data_dir": str(tmp_path)})

    df = pd.DataFrame({"Close": [1, 2]}, index=pd.date_range("2024-01-01", periods=2))
    path = tmp_path / "AAA.csv"
    df.to_csv(path, index_label="Date")

    called = {"value": False}

    def _build_abt(_frequency):
        called["value"] = True
        return {"AAA": path}

    build_abt_module = importlib.import_module("src.abt.build_abt")
    monkeypatch.setattr(build_abt_module, "build_abt", _build_abt)

    data = predict.load_prediction_data("daily")

    assert "AAA" in data
    assert not called["value"]
    assert list(data["AAA"].columns) == ["Close"]


def test_load_prediction_data_builds_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "CONFIG", {"etfs": ["AAA"], "data_dir": str(tmp_path)})

    built_path = tmp_path / "AAA.csv"

    def _build_abt(_frequency):
        df = pd.DataFrame({"Close": [3, 4]}, index=pd.date_range("2024-02-01", periods=2))
        df.to_csv(built_path, index_label="Date")
        return {"AAA": built_path}

    build_abt_module = importlib.import_module("src.abt.build_abt")
    monkeypatch.setattr(build_abt_module, "build_abt", _build_abt)

    data = predict.load_prediction_data("daily")

    assert "AAA" in data
    assert data["AAA"]["Close"].iloc[-1] == 4
