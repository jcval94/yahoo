import pandas as pd

import importlib

build_abt_module = importlib.import_module("src.abt.build_abt")


def test_build_abt_writes_incremental_combined_with_expected_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(build_abt_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        build_abt_module,
        "CONFIG",
        {
            "etfs": ["AAA", "BBB"],
            "start_date": "2020-01-01",
            "data_frequency": "1d",
            "include_prepost": False,
            "session_labels": {},
        },
    )

    idx = pd.date_range("2024-01-01", periods=3, freq="D")

    def _download_ticker(ticker, *_args, **_kwargs):
        offset = 0 if ticker == "AAA" else 10
        return pd.DataFrame(
            {
                "Open": [1 + offset, 2 + offset, 3 + offset],
                "High": [2 + offset, 3 + offset, 4 + offset],
                "Low": [0 + offset, 1 + offset, 2 + offset],
                "Close": [1.5 + offset, 2.5 + offset, 3.5 + offset],
                "Volume": [100, 200, 300],
            },
            index=idx,
        )

    def _enrich(df):
        out = df.copy()
        out["feature_x"] = out["Close"].pct_change().fillna(0)
        return out

    monkeypatch.setattr(build_abt_module, "download_ticker", _download_ticker)
    monkeypatch.setattr(build_abt_module, "enrich_indicators", _enrich)

    results = build_abt_module.build_abt("daily")

    assert set(results) == {"AAA", "BBB", "combined"}
    assert results["AAA"].exists()
    assert results["BBB"].exists()
    assert results["combined"].exists()

    combined = pd.read_csv(results["combined"])
    expected_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Ticker",
        "feature_x",
    ]
    assert list(combined.columns) == expected_columns
    assert len(combined) == 6
    assert set(combined["Ticker"]) == {"AAA", "BBB"}
