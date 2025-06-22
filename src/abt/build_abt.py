"""Build daily and weekly ABTs enriched with technical indicators."""
import logging
import yaml
from pathlib import Path
import pandas as pd
import yfinance as yf
import ta
import time
import socket
import requests_cache
from pandas_datareader.stooq import StooqDailyReader

from ..utils import (
    timed_stage,
    log_df_details,
    generate_sample_data,
    log_offline_mode,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# configuration lives at the project root two levels up from this file
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

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


def _download_yahoo(ticker, period, interval):
    """Download data using yfinance without passing a custom session."""
    return yf.download(
        ticker,
        period=period,
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
        period = "6mo"
        today = pd.Timestamp.today().normalize()
        start_dt = today - pd.DateOffset(months=6)

        if not _internet_ok():
            logger.warning("Runner sin internet. Usando datos simulados.")
            df = generate_sample_data(start_dt)
            log_df_details(f"downloaded {ticker}", df)
            return df

        df = pd.DataFrame()
        for attempt in range(1, retries + 1):
            try:
                df = _download_yahoo(ticker, period, interval)
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
    """Calculate basic technical indicators for a dataframe."""
    with timed_stage("indicator calculation"):
        if df.empty:
            logger.warning("DataFrame empty. Skipping indicators.")
            return df
        try:
            df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
            df["sma_20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
            df["sma_50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()
        except Exception:
            logger.exception("Failed to compute technical indicators")
            raise
    log_df_details("with indicators", df)
    return df

def build_abt() -> dict:
    """Build the analytic base table for all tickers defined in the config."""
    results = {}
    for ticker in CONFIG.get("etfs", []):
        try:
            with timed_stage(f"processing {ticker}"):
                df = download_ticker(ticker, CONFIG["start_date"])
                df = enrich_indicators(df)
                out_file = DATA_DIR / f"{ticker}.csv"
                df.index.name = "Date"
                df.to_csv(out_file, index_label="Date")
                log_df_details(f"saved {ticker}", df)
                results[ticker] = out_file
        except Exception:
            logger.error("Failed to process %s", ticker)
    log_offline_mode("build_abt")
    return results

if __name__ == "__main__":
    build_abt()
