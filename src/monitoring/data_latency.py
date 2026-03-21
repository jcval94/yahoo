"""Measure intraday data freshness across a short ticker set."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..abt.build_abt import CONFIG, download_ticker

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
DEFAULT_INTERVALS = ["1m", "5m"]
ROOT_DIR = Path(__file__).resolve().parents[2]
LATENCY_DIR = ROOT_DIR / "results" / "latency"


def _interval_to_minutes(interval: str) -> int | None:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    return None


def _is_intraday_granularity(index: pd.Index) -> bool:
    if len(index) < 1:
        return False
    parsed = pd.to_datetime(index, errors="coerce")
    parsed = parsed.dropna()
    if len(parsed) < 1:
        return False
    if len(parsed) == 1:
        ts = parsed[0]
        return bool(ts.hour or ts.minute or ts.second)

    series = pd.Series(parsed).sort_values()
    diffs = series.diff().dropna()
    if diffs.empty:
        return False
    min_step = diffs.min()
    return pd.Timedelta(0) < min_step <= pd.Timedelta(hours=12)


def _is_recent_intraday_bar(latency_minutes: float, interval: str) -> bool:
    interval_minutes = _interval_to_minutes(interval) or 60
    freshness_limit = max(interval_minutes * 3, 180)
    return latency_minutes <= freshness_limit


def _to_utc_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def measure_ticker_latency(ticker: str, intervals: list[str]) -> dict[str, object]:
    """Download intraday bars and return freshness metrics for one ticker."""
    download_time_utc = datetime.now(timezone.utc)

    for interval in intervals:
        try:
            frame = download_ticker(ticker=ticker, start="2024-01-01", interval=interval)
        except Exception as exc:  # pragma: no cover - network instability
            logger.warning("Latency download failed for %s (%s): %s", ticker, interval, exc)
            frame = pd.DataFrame()

        if frame.empty:
            continue

        last_bar_ts = _to_utc_timestamp(frame.index.max())
        latency_minutes = round((download_time_utc - last_bar_ts.to_pydatetime()).total_seconds() / 60.0, 2)
        granularity_ok = _is_intraday_granularity(frame.index)
        recent_ok = _is_recent_intraday_bar(latency_minutes, interval)
        if not granularity_ok:
            status = "invalid_granularity"
        elif not recent_ok:
            status = "stale_intraday"
        else:
            status = "ok"
        return {
            "ticker": ticker,
            "download_time_utc": download_time_utc.isoformat(),
            "last_bar_timestamp": last_bar_ts.isoformat(),
            "latency_minutes": latency_minutes,
            "interval_requested": interval,
            "interval_used": interval,
            "source": "yfinance",
            "granularity_ok": bool(granularity_ok),
            "status": status,
        }

    return {
        "ticker": ticker,
        "download_time_utc": download_time_utc.isoformat(),
        "last_bar_timestamp": "",
        "latency_minutes": "",
        "interval_requested": ",".join(intervals),
        "interval_used": "",
        "source": "yfinance",
        "granularity_ok": False,
        "status": "no_data",
    }


def build_latency_report(tickers: list[str], intervals: list[str]) -> Path:
    """Create a daily CSV with intraday latency per ticker."""
    LATENCY_DIR.mkdir(parents=True, exist_ok=True)

    rows = [measure_ticker_latency(ticker=ticker, intervals=intervals) for ticker in tickers]
    report_df = pd.DataFrame(rows)

    report_date = datetime.now(timezone.utc).date().isoformat()
    report_path = LATENCY_DIR / f"latency_{report_date}.csv"
    report_df.to_csv(report_path, index=False)
    logger.info("Latency report written to %s", report_path)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate intraday data latency report")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=CONFIG.get("etfs", DEFAULT_TICKERS)[:5],
        help="Ticker list to monitor (default: first 5 configured ETFs)",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=DEFAULT_INTERVALS,
        help="Ordered intraday intervals to try (default: 1m then 5m)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    build_latency_report(tickers=args.tickers, intervals=args.intervals)


if __name__ == "__main__":
    main()
