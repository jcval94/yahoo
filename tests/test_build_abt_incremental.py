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

    def fake_download(ticker, start, interval="1d", include_prepost=False, retries=3, **kwargs):
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


def test_build_abt_combined_from_saved_csvs_without_retaining_all_frames(tmp_path, monkeypatch):
    module = __import__("src.abt.build_abt", fromlist=["dummy"])

    monkeypatch.setattr(
        module,
        "CONFIG",
        {
            "etfs": ["AAA", "BBB"],
            "start_date": "2024-01-01",
            "data_frequency": "1d",
            "include_prepost": False,
            "data_dir": str(tmp_path),
        },
    )
    monkeypatch.setattr(module, "DATA_DIR", tmp_path)

    data_by_ticker = {
        "AAA": pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [1.0, 2.0],
                "Low": [1.0, 2.0],
                "Close": [1.0, 2.0],
                "Volume": [10, 10],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        ),
        "BBB": pd.DataFrame(
            {
                "Open": [3.0, 4.0],
                "High": [3.0, 4.0],
                "Low": [3.0, 4.0],
                "Close": [3.0, 4.0],
                "Volume": [20, 20],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-03"]),
        ),
    }

    def fake_download(ticker, start, interval="1d", include_prepost=False, retries=3, **kwargs):
        return data_by_ticker[ticker].copy()

    def fake_enrich(df):
        out = df.copy()
        out["feat"] = out["Close"] * 10
        return out

    monkeypatch.setattr(module, "download_ticker", fake_download)
    monkeypatch.setattr(module, "enrich_indicators", fake_enrich)

    read_calls = []
    original_read_csv = module.pd.read_csv

    def tracking_read_csv(*args, **kwargs):
        read_calls.append(str(args[0]))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(module.pd, "read_csv", tracking_read_csv)

    result = module.build_abt("daily", full_rebuild=True, safety_rows=2)
    combined = pd.read_csv(result["combined"], parse_dates=["Date"], index_col="Date")

    assert len(combined) == 4
    assert set(combined["Ticker"].unique()) == {"AAA", "BBB"}
    assert list(combined.index) == sorted(combined.index.tolist())
    assert any(path.endswith("AAA.csv") for path in read_calls)
    assert any(path.endswith("BBB.csv") for path in read_calls)


def test_download_ticker_offline_uses_long_history_sample(monkeypatch):
    module = __import__("src.abt.build_abt", fromlist=["dummy"])

    captured = {}

    monkeypatch.setattr(module, "_internet_ok", lambda: False)

    def fake_generate(start, periods=30):
        captured["periods"] = periods
        idx = pd.date_range("2024-01-01", periods=periods, freq="D")
        return pd.DataFrame(
            {
                "Open": [1.0] * periods,
                "High": [1.0] * periods,
                "Low": [1.0] * periods,
                "Close": [1.0] * periods,
                "Adj Close": [1.0] * periods,
                "Volume": [1000] * periods,
            },
            index=idx,
        )

    monkeypatch.setattr(module, "generate_sample_data", fake_generate)

    out = module.download_ticker("AAA", "2015-01-01", interval="1d")

    assert len(out) == captured["periods"]
    assert captured["periods"] >= 260


def test_download_ticker_intraday_uses_compatible_periods(monkeypatch):
    module = __import__("src.abt.build_abt", fromlist=["dummy"])
    monkeypatch.setattr(module, "_internet_ok", lambda: True)

    calls = []

    def fake_yf_download(*args, **kwargs):
        calls.append(kwargs)
        idx = pd.date_range("2026-01-01 09:30", periods=3, freq="1min")
        return pd.DataFrame(
            {
                "Open": [1.0, 1.0, 1.0],
                "High": [1.0, 1.0, 1.0],
                "Low": [1.0, 1.0, 1.0],
                "Close": [1.0, 1.0, 1.0],
                "Adj Close": [1.0, 1.0, 1.0],
                "Volume": [100, 100, 100],
            },
            index=idx,
        )

    monkeypatch.setattr(module.yf, "download", fake_yf_download)

    module.download_ticker("AAA", "2024-01-01", interval="1m")
    module.download_ticker("AAA", "2024-01-01", interval="5m")

    assert calls[0]["interval"] == "1m"
    assert calls[0]["period"] == "8d"
    assert calls[1]["interval"] == "5m"
    assert calls[1]["period"] == "60d"


def test_download_ticker_intraday_does_not_use_stooq_fallback(monkeypatch):
    module = __import__("src.abt.build_abt", fromlist=["dummy"])
    monkeypatch.setattr(module, "_internet_ok", lambda: True)
    monkeypatch.setattr(module.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    stooq_calls = {"count": 0}
    sample_calls = {"count": 0}

    def fake_stooq(*args, **kwargs):
        stooq_calls["count"] += 1
        return pd.DataFrame()

    def fake_sample(start, periods=30):
        sample_calls["count"] += 1
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "Open": [1.0] * 5,
                "High": [1.0] * 5,
                "Low": [1.0] * 5,
                "Close": [1.0] * 5,
                "Adj Close": [1.0] * 5,
                "Volume": [100] * 5,
            },
            index=idx,
        )

    monkeypatch.setattr(module, "_download_stooq", fake_stooq)
    monkeypatch.setattr(module, "generate_sample_data", fake_sample)

    out = module.download_ticker("AAA", "2024-01-01", interval="1m", retries=1)

    assert not out.empty
    assert stooq_calls["count"] == 0
    assert sample_calls["count"] == 1


def test_build_abt_full_daily_uses_batch_download(monkeypatch, tmp_path):
    module = __import__("src.abt.build_abt", fromlist=["dummy"])

    monkeypatch.setattr(
        module,
        "CONFIG",
        {
            "etfs": ["AAA", "BBB"],
            "start_date": "2024-01-01",
            "data_frequency": "1d",
            "include_prepost": False,
            "data_dir": str(tmp_path),
        },
    )
    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "_internet_ok", lambda: True)

    calls = {"batch": 0, "single": 0}

    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    batch = pd.DataFrame(
        {
            ("Open", "AAA"): [1.0, 2.0],
            ("High", "AAA"): [1.0, 2.0],
            ("Low", "AAA"): [1.0, 2.0],
            ("Close", "AAA"): [1.0, 2.0],
            ("Adj Close", "AAA"): [1.0, 2.0],
            ("Volume", "AAA"): [100, 100],
            ("Open", "BBB"): [3.0, 4.0],
            ("High", "BBB"): [3.0, 4.0],
            ("Low", "BBB"): [3.0, 4.0],
            ("Close", "BBB"): [3.0, 4.0],
            ("Adj Close", "BBB"): [3.0, 4.0],
            ("Volume", "BBB"): [200, 200],
        },
        index=idx,
    )
    batch.columns = pd.MultiIndex.from_tuples(batch.columns)

    def fake_yf_download(*args, **kwargs):
        calls["batch"] += 1
        return batch

    def fake_single(*args, **kwargs):
        calls["single"] += 1
        return pd.DataFrame()

    monkeypatch.setattr(module.yf, "download", fake_yf_download)
    monkeypatch.setattr(module, "download_ticker", fake_single)
    monkeypatch.setattr(module, "enrich_indicators", lambda df: df)

    out = module.build_abt("daily", full_rebuild=True)

    assert calls["batch"] == 1
    assert calls["single"] == 0
    assert "AAA" in out and "BBB" in out and "combined" in out


def test_build_abt_skips_writing_empty_ticker_csv(tmp_path, monkeypatch):
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
    monkeypatch.setattr(module, "_internet_ok", lambda: True)
    monkeypatch.setattr(module, "download_ticker", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(module, "enrich_indicators", lambda df: df)

    out = module.build_abt("daily", full_rebuild=True)

    assert "AAA" not in out
    assert not (tmp_path / "AAA.csv").exists()
