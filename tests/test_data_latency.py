from datetime import datetime, timezone

import pandas as pd

import src.monitoring.data_latency as latency


class _FrozenDateTime:
    @classmethod
    def now(cls, tz=None):
        base = datetime(2026, 3, 21, 15, 0, tzinfo=timezone.utc)
        if tz is None:
            return base
        return base.astimezone(tz)


def test_measure_ticker_latency_marks_invalid_granularity(monkeypatch):
    idx = pd.date_range("2026-03-18", periods=3, freq="D")
    frame = pd.DataFrame({"Close": [1.0, 1.0, 1.0]}, index=idx)

    monkeypatch.setattr(latency, "datetime", _FrozenDateTime)
    monkeypatch.setattr(latency, "download_ticker", lambda **kwargs: frame)

    out = latency.measure_ticker_latency("AAA", ["1m"])

    assert out["interval_requested"] == "1m"
    assert out["interval_used"] == "1m"
    assert out["source"] == "yfinance"
    assert out["granularity_ok"] is False
    assert out["status"] == "invalid_granularity"


def test_measure_ticker_latency_requires_recent_intraday_bar(monkeypatch):
    idx = pd.date_range("2026-03-21 13:00", periods=3, freq="1h", tz="UTC")
    frame = pd.DataFrame({"Close": [1.0, 1.0, 1.0]}, index=idx)

    monkeypatch.setattr(latency, "datetime", _FrozenDateTime)
    monkeypatch.setattr(latency, "download_ticker", lambda **kwargs: frame)

    out = latency.measure_ticker_latency("AAA", ["1h"])

    assert out["granularity_ok"] is True
    assert out["status"] == "ok"
