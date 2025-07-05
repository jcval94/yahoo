"""Data extraction and preprocessing utilities."""
import logging
from typing import Dict, List

import pandas as pd
import yfinance as yf

from .utils import (
    timed_stage,
    log_df_details,
    generate_sample_data,
    log_offline_mode,
)

logger = logging.getLogger(__name__)


def extract_data(tickers: List[str], start: str) -> Dict[str, pd.DataFrame]:
    """Download raw price data for a list of tickers."""
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    data = {}
    for t in tickers:
        with timed_stage(f"download {t}"):
            try:
                # yfinance 0.2.37 changed auto_adjust default to True.
                # Force False to maintain the schema used during training.
                df = yf.download(t, start=start, progress=False, auto_adjust=False)
            except Exception:
                logger.exception("Failed to download %s, using sample", t)
                df = generate_sample_data(start)
        if df.empty:
            logger.warning("%s download empty, using sample", t)
            df = generate_sample_data(start)
        log_df_details(f"raw {t}", df)
        data[t] = df
    log_offline_mode("extract_data")
    return data


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators."""
    if df.empty:
        log_df_details("enrich empty", df)
        return df
    from .features import add_technical_indicators

    df = add_technical_indicators(df)
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
    log_offline_mode("preprocess_data")
    return processed
