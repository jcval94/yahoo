import pandas as pd

from src.abt.build_abt import build_abt
from src.features import get_feature_recalc_rows


def test_get_feature_recalc_rows_respects_minimum_window():
    recalc_rows, window_max = get_feature_recalc_rows(10)
    assert window_max == 90
    assert recalc_rows >= 91

    recalc_rows, window_max = get_feature_recalc_rows(250)
    assert window_max == 90
    assert recalc_rows == 250


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


def test_build_abt_incremental_safety_rows_keeps_latest_rows_equal_to_full_rebuild(tmp_path, monkeypatch):
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

    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    base = pd.DataFrame(
        {
            "Open": range(1, 41),
            "High": range(1, 41),
            "Low": range(1, 41),
            "Close": range(1, 41),
            "Volume": [100] * 40,
            "Ticker": ["AAA"] * 40,
        },
        index=idx,
    )

    out_file = tmp_path / "AAA.csv"
    base.iloc[:30].to_csv(out_file, index_label="Date")

    def fake_download(ticker, start, interval="1d", include_prepost=False, retries=3):
        return base.loc[base.index >= pd.Timestamp(start)].copy()

    def fake_enrich(df):
        out = df.copy()
        out["feat_lag_3"] = out["Close"].shift(3)
        out["feat_roll_5"] = out["Close"].rolling(5, min_periods=1).mean()
        return out

    monkeypatch.setattr(module, "download_ticker", fake_download)
    monkeypatch.setattr(module, "enrich_indicators", fake_enrich)

    small_tail_result = build_abt("daily", full_rebuild=False, safety_rows=5)
    small_tail_df = pd.read_csv(small_tail_result["AAA"], parse_dates=["Date"], index_col="Date")

    full_result = build_abt("daily", full_rebuild=True, safety_rows=5)
    full_df = pd.read_csv(full_result["AAA"], parse_dates=["Date"], index_col="Date")

    n_latest = 10
    pd.testing.assert_frame_equal(small_tail_df.tail(n_latest), full_df.tail(n_latest), check_dtype=False)


def test_build_abt_uses_frequency_specific_safety_rows_from_config(tmp_path, monkeypatch):
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
            "recalc": {
                "min_safety_rows": 180,
                "safety_rows_by_frequency": {"weekly": 77},
            },
        },
    )
    monkeypatch.setattr(module, "DATA_DIR", tmp_path)

    idx = pd.date_range("2024-01-01", periods=5, freq="D")

    def fake_download(ticker, start, interval="1d", include_prepost=False, retries=3):
        return pd.DataFrame(
            {
                "Open": [1, 2, 3, 4, 5],
                "High": [1, 2, 3, 4, 5],
                "Low": [1, 2, 3, 4, 5],
                "Close": [1, 2, 3, 4, 5],
                "Volume": [10, 10, 10, 10, 10],
            },
            index=idx,
        )

    called = {"safety_rows": None}

    def fake_recalc_rows(safety_rows, safety_margin=1):
        called["safety_rows"] = safety_rows
        return max(safety_rows, 91), 90

    monkeypatch.setattr(module, "download_ticker", fake_download)
    monkeypatch.setattr(module, "enrich_indicators", lambda df: df)
    monkeypatch.setattr(module, "get_feature_recalc_rows", fake_recalc_rows)

    build_abt("weekly", full_rebuild=True)

    assert called["safety_rows"] == 77
