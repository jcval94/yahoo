"""Build analytic base tables enriched with technical indicators."""
import logging
from pathlib import Path
import pandas as pd
import yfinance as yf
import time
import socket
import requests_cache
from pandas_datareader.stooq import StooqDailyReader

from ..utils import (
    timed_stage,
    log_df_details,
    generate_sample_data,
    fallback_periods_from_start,
    log_offline_mode,
    load_config,
)
from ..features import get_feature_recalc_rows

logger = logging.getLogger(__name__)

# configuration lives at the project root two levels up from this file
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
CONFIG = load_config(CONFIG_PATH)

# store data in the directory defined in config at the project root
DATA_DIR = Path(__file__).resolve().parents[2] / CONFIG.get("data_dir", "data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

CACHE_EXPIRE = 60 * 60  # 1 h

# Install a global cache for all requests instead of passing a session to
# yfinance. The library now manages its own session, so this avoids
# compatibility issues and still provides caching.
requests_cache.install_cache("yf_cache", expire_after=CACHE_EXPIRE)


def _internet_ok(host="query1.finance.yahoo.com", port=443, timeout=3):
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        return False


def _download_yahoo(ticker, start, end, interval):
    """Download data using yfinance without passing a custom session."""
    return yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        threads=False,
        auto_adjust=False,
    )


def _download_stooq(ticker, start, end):
    return StooqDailyReader(ticker, start=start, end=end).read()


def _extract_batch_ticker_frame(batch_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract a single ticker frame from a multi-ticker yfinance payload."""
    if batch_df.empty:
        return pd.DataFrame()

    if not isinstance(batch_df.columns, pd.MultiIndex):
        return batch_df.copy()

    out = pd.DataFrame(index=batch_df.index)

    if ticker in batch_df.columns.get_level_values(0):
        subset = batch_df[ticker]
        out = subset.copy()
    elif ticker in batch_df.columns.get_level_values(1):
        for field in batch_df.columns.get_level_values(0).unique():
            key = (field, ticker)
            if key in batch_df.columns:
                out[field] = batch_df[key]

    if out.empty:
        return out
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    return out


def _download_daily_batch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download daily data for many tickers in one Yahoo request."""
    if not tickers:
        return {}
    try:
        batch_df = yf.download(
            tickers,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=False,
            group_by="column",
        )
    except Exception:
        logger.exception("Batch download failed for %s tickers", len(tickers))
        return {}

    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        part = _extract_batch_ticker_frame(batch_df, ticker)
        if part.empty:
            continue
        out[ticker] = part.sort_index()
    return out


def _is_intraday_interval(interval: str) -> bool:
    return interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


def _intraday_period_for_interval(interval: str) -> str:
    """Return a Yahoo-compatible period for each intraday interval."""
    if interval == "1m":
        return "8d"
    if interval in {"2m", "5m"}:
        return "60d"
    return "730d"


def _normalize_ts(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def _session_from_timestamp(ts: pd.Timestamp, labels: dict | None = None) -> str:
    labels = labels or {}
    label_open = labels.get("open", "open")
    label_midday = labels.get("midday", "midday")
    label_close = labels.get("close", "close")
    label_ah = labels.get("after_hours", "after_hours")
    label_pre = labels.get("premarket", label_ah)

    ts_ny = _normalize_ts(pd.Timestamp(ts))
    minute_of_day = ts_ny.hour * 60 + ts_ny.minute

    if minute_of_day < 570:
        return label_pre
    if minute_of_day < 690:
        return label_open
    if minute_of_day < 900:
        return label_midday
    if minute_of_day <= 960:
        return label_close
    return label_ah


def add_session_column(df: pd.DataFrame, labels: dict | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["session"] = [
        _session_from_timestamp(ts, labels) for ts in out.index
    ]
    return out

def download_ticker(
    ticker: str,
    start: str,
    interval: str = "1d",
    include_prepost: bool = False,
    retries: int = 3,
    internet_available: bool | None = None,
) -> pd.DataFrame:
    """Download historical data with fallbacks for CI environments."""
    with timed_stage(f"download {ticker}"):
        today = pd.Timestamp.today().normalize()
        start_dt = pd.to_datetime(start)

        if internet_available is None:
            internet_available = _internet_ok()

        if not internet_available:
            logger.warning("Runner sin internet. Usando datos simulados.")
            periods = fallback_periods_from_start(start_dt)
            df = generate_sample_data(start_dt, periods=periods)
            log_df_details(f"downloaded {ticker}", df)
            return df

        df = pd.DataFrame()
        for attempt in range(1, retries + 1):
            try:
                if _is_intraday_interval(interval):
                    period = _intraday_period_for_interval(interval)
                    df = yf.download(
                        ticker,
                        period=period,
                        interval=interval,
                        prepost=include_prepost,
                        progress=False,
                        threads=False,
                        auto_adjust=False,
                    )
                else:
                    df = _download_yahoo(
                        ticker,
                        start_dt.strftime("%Y-%m-%d"),
                        today.strftime("%Y-%m-%d"),
                        interval,
                    )
                if not df.empty:
                    break
            except Exception as exc:
                logger.exception("yf intento %d falló: %s", attempt, exc)
                df = pd.DataFrame()
            time.sleep(1)

        if df.empty and not _is_intraday_interval(interval):
            try:
                df = _download_stooq(ticker, start_dt, today)
                logger.info("Stooq suministró %d filas para %s", len(df), ticker)
            except Exception as exc:
                logger.exception("Stooq también falló: %s", exc)

        if df.empty:
            logger.warning("Todo falló. Generando datos de ejemplo.")
            periods = fallback_periods_from_start(start_dt)
            df = generate_sample_data(start_dt, periods=periods)

        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)

        # ensure rows are in chronological order for rolling features
        df = df.sort_index()

    log_df_details(f"downloaded {ticker}", df)
    return df

def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for a dataframe."""
    with timed_stage("indicator calculation"):
        if df.empty:
            logger.warning("DataFrame empty. Skipping indicators.")
            return df
        try:
            from ..features import add_technical_indicators

            df = add_technical_indicators(df)
        except Exception:
            logger.exception("Failed to compute technical indicators")
            raise
    log_df_details("with indicators", df)
    return df


def _to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLC data to weekly frequency."""
    if df.empty:
        return df

    agg = {}
    for col in df.columns:
        if col == "Open":
            agg[col] = "first"
        elif col == "High":
            agg[col] = "max"
        elif col == "Low":
            agg[col] = "min"
        elif col in {"Close", "Adj Close"}:
            # use the mean close to approximate the average price of the week
            agg[col] = "mean"
        elif col == "Volume":
            agg[col] = "sum"
        else:
            agg[col] = "last"

    # Use Sunday as the label for each aggregated week
    weekly = df.resample("W-SUN").agg(agg)
    return weekly


def _to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLC data to monthly frequency."""
    if df.empty:
        return df

    agg = {}
    for col in df.columns:
        if col == "Open":
            agg[col] = "first"
        elif col == "High":
            agg[col] = "max"
        elif col == "Low":
            agg[col] = "min"
        elif col in {"Close", "Adj Close"}:
            agg[col] = "last"
        elif col == "Volume":
            agg[col] = "sum"
        else:
            agg[col] = "last"

    monthly = df.resample("M").agg(agg)
    return monthly



def _output_file_for_ticker(ticker: str, frequency: str) -> Path:
    if frequency == "intraday":
        return DATA_DIR / f"{ticker}_intraday.csv"
    suffix = "" if frequency == "daily" else f"_{frequency}"
    return DATA_DIR / f"{ticker}{suffix}.csv"


def _combined_output_file(frequency: str) -> Path:
    if frequency == "intraday":
        return DATA_DIR / "etfs_combined_intraday.csv"
    suffix = "" if frequency == "daily" else f"_{frequency}"
    return DATA_DIR / f"etfs_combined{suffix}.csv"


def _build_combined_from_ticker_files(ticker_files: list[Path], combined_file: Path) -> pd.DataFrame:
    """Build the combined ABT by reading each ticker CSV from disk."""
    combined_parts = []
    for ticker_file in ticker_files:
        if not ticker_file.exists():
            continue
        part = pd.read_csv(ticker_file, parse_dates=["Date"], index_col="Date")
        combined_parts.append(part)

    if not combined_parts:
        return pd.DataFrame()

    combined_processed = pd.concat(combined_parts).sort_index()
    combined_processed.index.name = "Date"
    combined_processed.to_csv(combined_file, index_label="Date")
    return combined_processed


def _read_existing_abt(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        return df.sort_index()
    except Exception:
        logger.exception("No se pudo leer ABT previo: %s", path)
        return pd.DataFrame()


def _apply_frequency(df: pd.DataFrame, frequency: str, session_labels: dict) -> pd.DataFrame:
    if frequency == "weekly":
        return _to_weekly(df)
    if frequency == "monthly":
        return _to_monthly(df)
    if frequency == "intraday":
        return add_session_column(df, session_labels)
    return df


def _new_data_start(last_ts: pd.Timestamp, frequency: str) -> str:
    if frequency == "intraday":
        next_ts = last_ts + pd.Timedelta(minutes=1)
    elif frequency == "weekly":
        next_ts = last_ts + pd.Timedelta(days=7)
    elif frequency == "monthly":
        next_ts = last_ts + pd.offsets.MonthBegin(1)
    else:
        next_ts = last_ts + pd.Timedelta(days=1)
    return pd.Timestamp(next_ts).strftime("%Y-%m-%d")

def build_abt(frequency: str = "daily", full_rebuild: bool = False, safety_rows: int = 180) -> dict:
    """Build analytic base tables for all tickers defined in the config."""
    results = {}

    configured_interval = CONFIG.get("data_frequency", "1d")
    include_prepost = bool(CONFIG.get("include_prepost", False))
    session_labels = CONFIG.get("session_labels", {})

    if frequency in {"daily", "weekly", "monthly"}:
        interval = "1d"
    else:
        interval = configured_interval
        frequency = "intraday"

    five_years_ago = pd.Timestamp.today().normalize() - pd.DateOffset(years=5)
    config_start = pd.to_datetime(CONFIG["start_date"])
    start_dt = max(config_start, five_years_ago)

    recalc_rows = get_feature_recalc_rows(safety_rows)
    internet_available = _internet_ok()

    batch_daily_frames: dict[str, pd.DataFrame] = {}
    if full_rebuild and frequency == "daily" and internet_available:
        with timed_stage("download batch daily tickers"):
            batch_daily_frames = _download_daily_batch(
                list(CONFIG.get("etfs", [])),
                start_dt.strftime("%Y-%m-%d"),
                pd.Timestamp.today().normalize().strftime("%Y-%m-%d"),
            )

    ticker_output_files = []
    for ticker in CONFIG.get("etfs", []):
        out_file = _output_file_for_ticker(ticker, frequency)
        existing_df = pd.DataFrame() if full_rebuild else _read_existing_abt(out_file)

        download_start = start_dt.strftime("%Y-%m-%d")
        if not full_rebuild and not existing_df.empty:
            last_ts = pd.Timestamp(existing_df.index.max())
            download_start = _new_data_start(last_ts, frequency)

        try:
            with timed_stage(f"download {ticker}"):
                fresh_df = batch_daily_frames.get(ticker, pd.DataFrame())
                if fresh_df.empty:
                    fresh_df = download_ticker(
                        ticker,
                        download_start,
                        interval=interval,
                        include_prepost=include_prepost,
                        internet_available=internet_available,
                    )
                fresh_df = _apply_frequency(fresh_df, frequency, session_labels)
                fresh_df["Ticker"] = ticker
        except Exception:
            logger.exception("Failed to download %s", ticker)
            continue

        if not full_rebuild and not existing_df.empty:
            last_ts = pd.Timestamp(existing_df.index.max())
            fresh_df = fresh_df[fresh_df.index > last_ts]

        if fresh_df.empty and not full_rebuild and not existing_df.empty:
            logger.info("Sin filas nuevas para %s. Se conserva ABT existente.", ticker)
            final_df = existing_df
        else:
            history_tail = pd.DataFrame() if full_rebuild else existing_df.tail(recalc_rows)
            base_df = pd.concat([history_tail, fresh_df]).sort_index()
            base_df = base_df[~base_df.index.duplicated(keep="last")]
            with timed_stage(f"processing {ticker}"):
                recalculated = enrich_indicators(base_df)

            if full_rebuild or existing_df.empty:
                final_df = recalculated
            else:
                historical_keep = existing_df.iloc[:-len(history_tail)] if len(history_tail) else existing_df
                final_df = pd.concat([historical_keep, recalculated]).sort_index()
                final_df = final_df[~final_df.index.duplicated(keep="last")]

        if final_df.empty or len(final_df.columns) == 0:
            logger.error(
                "ABT vacío para %s (%s). Se omite escritura para evitar CSV inválido.",
                ticker,
                frequency,
            )
            if not existing_df.empty:
                logger.info("Se conserva ABT previo no vacío para %s", ticker)
                results[ticker] = out_file
                ticker_output_files.append(out_file)
            continue

        final_df.index.name = "Date"
        final_df.to_csv(out_file, index_label="Date")
        log_df_details(f"saved {out_file.stem}", final_df)
        results[ticker] = out_file
        ticker_output_files.append(out_file)

    if not ticker_output_files:
        log_offline_mode(f"build_{frequency}_abt")
        return results

    combined_file = _combined_output_file(frequency)
    combined_processed = _build_combined_from_ticker_files(ticker_output_files, combined_file)
    if combined_processed.empty:
        log_offline_mode(f"build_{frequency}_abt")
        return results

    log_df_details(f"saved {combined_file.stem}", combined_processed)
    results["combined"] = combined_file

    log_offline_mode(f"build_{frequency}_abt")
    return results


def build_weekly_abt(full_rebuild: bool = False, safety_rows: int = 180) -> dict:
    """Build weekly analytic base tables for configured tickers."""
    return build_abt("weekly", full_rebuild=full_rebuild, safety_rows=safety_rows)


def build_monthly_abt(full_rebuild: bool = False, safety_rows: int = 180) -> dict:
    """Build monthly analytic base tables for configured tickers."""
    return build_abt("monthly", full_rebuild=full_rebuild, safety_rows=safety_rows)

def main():
    """Entry point for command line execution."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    import argparse

    parser = argparse.ArgumentParser(description="Build analytic base tables")
    parser.add_argument(
        "--frequency",
        choices=["daily", "weekly", "monthly", "intraday"],
        default="daily",
        help="data frequency for the ABT",
    )
    parser.add_argument("--weekly", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--monthly", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="recalcula el ABT completo desde la fecha de inicio",
    )
    parser.add_argument(
        "--safety-rows",
        type=int,
        default=180,
        help="cola de seguridad de filas para recalcular features rolling/lags",
    )
    args = parser.parse_args()

    if args.weekly:
        args.frequency = "weekly"
    elif args.monthly:
        args.frequency = "monthly"

    build_abt(args.frequency, full_rebuild=args.full_rebuild, safety_rows=args.safety_rows)


if __name__ == "__main__":
    main()
