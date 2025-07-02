"""Build analytic base tables enriched with technical indicators."""
import logging
from pathlib import Path
from ..utils import load_config
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
    log_offline_mode,
    load_config,
)

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
    )


def _download_stooq(ticker, start, end):
    return StooqDailyReader(ticker, start=start, end=end).read()

def download_ticker(
    ticker: str,
    start: str,
    interval: str = "1d",
    retries: int = 3,
) -> pd.DataFrame:
    """Download historical data with fallbacks for CI environments."""
    with timed_stage(f"download {ticker}"):
        today = pd.Timestamp.today().normalize()
        start_dt = pd.to_datetime(start)

        if not _internet_ok():
            logger.warning("Runner sin internet. Usando datos simulados.")
            df = generate_sample_data(start_dt)
            log_df_details(f"downloaded {ticker}", df)
            return df

        df = pd.DataFrame()
        for attempt in range(1, retries + 1):
            try:
                df = _download_yahoo(
                    ticker,
                    start_dt.strftime("%Y-%m-%d"),
                    today.strftime("%Y-%m-%d"),
                    interval,
                )
                if not df.empty:
                    break
            except Exception as exc:
                logger.error("yf intento %d falló: %s", attempt, exc)
                df = pd.DataFrame()
            time.sleep(1)

        if df.empty:
            try:
                df = _download_stooq(ticker, start_dt, today)
                logger.info("Stooq suministró %d filas para %s", len(df), ticker)
            except Exception as exc:
                logger.error("Stooq también falló: %s", exc)

        if df.empty:
            logger.warning("Todo falló. Generando datos de ejemplo.")
            df = generate_sample_data(start_dt)

        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)

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
            agg[col] = "last"
        elif col == "Volume":
            agg[col] = "sum"
        else:
            agg[col] = "last"

    weekly = df.resample("W").agg(agg)
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

def build_abt(frequency: str = "daily") -> dict:
    """Build analytic base tables for all tickers defined in the config."""
    results = {}
    combined_frames = []

    five_years_ago = pd.Timestamp.today().normalize() - pd.DateOffset(years=5)
    config_start = pd.to_datetime(CONFIG["start_date"])
    start_dt = max(config_start, five_years_ago)

    for ticker in CONFIG.get("etfs", []):
        try:
            with timed_stage(f"download {ticker}"):
                df = download_ticker(ticker, start_dt.strftime("%Y-%m-%d"))
                if frequency == "weekly":
                    df = _to_weekly(df)
                elif frequency == "monthly":
                    df = _to_monthly(df)
                df["Ticker"] = ticker
                combined_frames.append(df)
        except Exception:
            logger.error("Failed to download %s", ticker)

    if not combined_frames:
        log_offline_mode(f"build_{frequency}_abt")
        return results

    combined_df = pd.concat(combined_frames)

    processed_frames = []
    for ticker, group_df in combined_df.groupby("Ticker"):
        with timed_stage(f"processing {ticker}"):
            group_df = enrich_indicators(group_df)
        suffix = "" if frequency == "daily" else f"_{frequency}"
        out_file = DATA_DIR / f"{ticker}{suffix}.csv"
        group_df.index.name = "Date"
        group_df.to_csv(out_file, index_label="Date")
        log_df_details(f"saved {ticker}{suffix}", group_df)
        results[ticker] = out_file
        processed_frames.append(group_df)

    combined_processed = pd.concat(processed_frames)
    suffix = "" if frequency == "daily" else f"_{frequency}"
    combined_file = DATA_DIR / f"etfs_combined{suffix}.csv"
    combined_processed.index.name = "Date"
    combined_processed.to_csv(combined_file, index_label="Date")
    log_df_details(f"saved combined{suffix}", combined_processed)
    results["combined"] = combined_file

    log_offline_mode(f"build_{frequency}_abt")
    return results


def build_weekly_abt() -> dict:
    """Build weekly analytic base tables for configured tickers."""
    return build_abt("weekly")


def build_monthly_abt() -> dict:
    """Build monthly analytic base tables for configured tickers."""
    return build_abt("monthly")

def main():
    """Entry point for command line execution."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    import argparse

    parser = argparse.ArgumentParser(description="Build analytic base tables")
    parser.add_argument(
        "--frequency",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="data frequency for the ABT",
    )
    parser.add_argument("--weekly", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--monthly", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.weekly:
        args.frequency = "weekly"
    elif args.monthly:
        args.frequency = "monthly"

    build_abt(args.frequency)


if __name__ == "__main__":
    main()
