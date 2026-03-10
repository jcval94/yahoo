import pandas as pd

from src.abt.build_abt import build_abt
from src.features import get_feature_recalc_rows


def test_get_feature_recalc_rows_respects_minimum_window():
    assert get_feature_recalc_rows(10) >= 91
    assert get_feature_recalc_rows(250) == 250


def test_build_abt_incremental_recomputes_tail_and_appends_new_rows(tmp_path, monkeypatch):
    module = __import__("src.abt.build_abt", fromlist=["dummy"])

    monkeypatch.setattr(
        module,
        "CONFIG",
        {
            "etfs": ["AAA"],
            "start_date": "2024-01-01",
            "data_frequency": "1d",
            "include_prepost": False,
            "data_dir": str(tmp_path),
        },
    )
    monkeypatch.setattr(module, "DATA_DIR", tmp_path)

    idx_existing = pd.date_range("2024-01-01", periods=5, freq="D")
    existing = pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [1, 2, 3, 4, 5],
            "Low": [1, 2, 3, 4, 5],
            "Close": [1, 2, 3, 4, 5],
            "Volume": [10, 10, 10, 10, 10],
            "Ticker": ["AAA"] * 5,
            "legacy": [100, 101, 102, 103, 104],
        },
        index=idx_existing,
    )
    out_file = tmp_path / "AAA.csv"
    existing.to_csv(out_file, index_label="Date")

    calls = {"start": None}

    def fake_download(ticker, start, interval="1d", include_prepost=False, retries=3):
        calls["start"] = start
        idx_new = pd.date_range("2024-01-06", periods=2, freq="D")
        return pd.DataFrame(
            {
                "Open": [6, 7],
                "High": [6, 7],
                "Low": [6, 7],
                "Close": [6, 7],
                "Volume": [10, 10],
            },
            index=idx_new,
        )

    def fake_enrich(df):
        out = df.copy()
        out["feat"] = out["Close"] * 10
        return out

    monkeypatch.setattr(module, "download_ticker", fake_download)
    monkeypatch.setattr(module, "enrich_indicators", fake_enrich)

    result = build_abt("daily", full_rebuild=False, safety_rows=2)
    final_df = pd.read_csv(result["AAA"], parse_dates=["Date"], index_col="Date")

    assert calls["start"] == "2024-01-06"
    assert len(final_df) == 7
    assert final_df.index.is_unique
    assert final_df.loc[pd.Timestamp("2024-01-02"), "legacy"] == 101
    assert final_df.loc[pd.Timestamp("2024-01-07"), "feat"] == 70
