"""Data extraction and preprocessing utilities."""
import logging
from typing import Dict, List

import pandas as pd
import ta
import yfinance as yf

from .utils import timed_stage, log_df_details

logger = logging.getLogger(__name__)


def extract_data(tickers: List[str], start: str) -> Dict[str, pd.DataFrame]:
    """Download raw price data for a list of tickers."""
    data = {}
    for t in tickers:
        with timed_stage(f"download {t}"):
            df = yf.download(t, start=start, progress=False)
        log_df_details(f"raw {t}", df)
        data[t] = df
    return data


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators."""
    if df.empty:
        log_df_details("enrich empty", df)
        return df
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["sma_20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()
    log_df_details("enriched", df)
    return df


def preprocess_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Apply feature enrichment to all dataframes."""
    processed = {}
    for t, df in data.items():
        with timed_stage(f"preprocess {t}"):
            processed_df = enrich_features(df)
        log_df_details(f"processed {t}", processed_df)
        processed[t] = processed_df
    return processed
