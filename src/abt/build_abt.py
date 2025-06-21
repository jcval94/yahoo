"""Build daily and weekly ABTs enriched with technical indicators."""
import logging
import yaml
from pathlib import Path
import pandas as pd
import yfinance as yf
import ta

from ..utils import timed_stage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

with open(CONFIG_PATH) as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

def download_ticker(ticker: str, start: str) -> pd.DataFrame:
    """Download historical data for a single ticker."""
    with timed_stage(f"download {ticker}"):
        df = yf.download(ticker, start=start)
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
                df.to_csv(out_file)
                results[ticker] = out_file
        except Exception:
            logger.error("Failed to process %s", ticker)
    return results

if __name__ == "__main__":
    build_abt()
