"""Load and align exogenous macro/risk factors."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


DEFAULT_EXOGENOUS = {
    "vix": "^VIX",
    "credit_spread_proxy": "HYG",
}


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted by a timezone-naive datetime index."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)
    out = out.sort_index()
    return out


def _empty_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    return pd.DataFrame(index=idx)


def load_market_series(
    symbol: str,
    column_name: str,
    *,
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Load a market proxy with yfinance using close prices."""
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame(columns=[column_name])

    if df.empty:
        return pd.DataFrame(columns=[column_name])

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if close_col not in df.columns:
        return pd.DataFrame(columns=[column_name])

    series_df = df[[close_col]].rename(columns={close_col: column_name})
    return _normalize_index(series_df)


def load_fomc_calendar(
    *,
    start: str,
    end: str | None = None,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load FOMC event dates and expose a binary event feature."""
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end) if end else pd.Timestamp.today().normalize()

    if csv_path is not None and Path(csv_path).exists():
        cal_df = pd.read_csv(csv_path)
        if "date" not in cal_df.columns:
            raise ValueError("FOMC calendar CSV must contain a 'date' column")
        event_dates = pd.to_datetime(cal_df["date"], errors="coerce").dropna().dt.normalize()
    else:
        # Minimal built-in fallback schedule for offline/test environments.
        fallback_dates = [
            "2023-01-31", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
            "2023-09-20", "2023-11-01", "2023-12-13", "2024-01-31", "2024-03-20",
            "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07",
            "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
            "2025-07-30", "2025-09-17", "2025-11-06", "2025-12-17", "2026-01-28",
        ]
        event_dates = pd.to_datetime(pd.Series(fallback_dates)).dt.normalize()

    event_dates = event_dates[(event_dates >= start_ts.normalize()) & (event_dates <= end_ts.normalize())]
    if event_dates.empty:
        idx = pd.date_range(start=start_ts.normalize(), end=end_ts.normalize(), freq="D")
        return pd.DataFrame({"is_fomc_day": 0}, index=idx)

    idx = pd.date_range(start=start_ts.normalize(), end=end_ts.normalize(), freq="D")
    out = pd.DataFrame(index=idx)
    out["is_fomc_day"] = out.index.normalize().isin(set(event_dates)).astype(int)
    return out


def load_exogenous_factors(
    config: dict,
    *,
    index: Iterable[pd.Timestamp],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Load configured exogenous factors and align to a target index."""
    exogenous_cfg = config.get("exogenous", {}) if isinstance(config, dict) else {}
    enabled = bool(exogenous_cfg.get("enabled", False))
    if not enabled:
        return _empty_frame(pd.DatetimeIndex(index))

    index_df = _empty_frame(pd.DatetimeIndex(index))
    factors: list[pd.DataFrame] = []

    if exogenous_cfg.get("use_vix", False):
        factors.append(load_market_series("^VIX", "vix_close", start=start, end=end))

    if exogenous_cfg.get("use_credit_spread_proxy", False):
        symbol = exogenous_cfg.get("credit_proxy_symbol", DEFAULT_EXOGENOUS["credit_spread_proxy"])
        factors.append(load_market_series(symbol, "credit_spread_proxy_close", start=start, end=end))

    if exogenous_cfg.get("use_fomc_calendar", False):
        factors.append(
            load_fomc_calendar(
                start=start,
                end=end,
                csv_path=exogenous_cfg.get("fomc_calendar_path"),
            )
        )

    if not factors:
        return index_df

    merged = pd.concat([index_df] + factors, axis=1)
    merged = merged[~merged.index.duplicated(keep="last")]
    return _normalize_index(merged)
