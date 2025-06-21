"""Build daily and weekly ABTs enriched with technical indicators."""
import logging
import yaml
from pathlib import Path
import pandas as pd
import yfinance as yf
import ta
import time

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

def download_ticker(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
    retries: int = 3,
) -> pd.DataFrame:
    """Download historical data or fall back to generated sample data."""
    with timed_stage(f"download {ticker}"):
        logger.info(
            "yf.download params ticker=%s start=%s end=%s interval=%s",
            ticker,
            start,
            end or "today",
            interval,
        )
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) if end else None
        df = pd.DataFrame()
        for attempt in range(1, retries + 1):
            try:
                df = yf.download(
                    ticker,
                    start=start_dt,
                    end=end_dt,
                    interval=interval,
                    progress=False,
                    threads=False,
                )
            except Exception as exc:
                logger.error(
                    "Attempt %d to download %s failed: %s", attempt, ticker, exc
                )
                df = pd.DataFrame()
            if not df.empty:
                break
            if attempt < retries:
                time.sleep(1)
        if df.empty:
            logger.warning("%s download empty, using sample data", ticker)
            df = generate_sample_data(start)
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
