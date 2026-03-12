"""Data extraction and preprocessing utilities."""
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

from .data.exogenous import load_exogenous_factors
from .utils import (
    timed_stage,
    log_df_details,
    generate_sample_data,
    fallback_periods_from_start,
    log_offline_mode,
    load_config,
)

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
CONFIG = load_config(CONFIG_PATH)


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
                df = generate_sample_data(start, periods=fallback_periods_from_start(start))
        if df.empty:
            logger.warning("%s download empty, using sample", t)
            df = generate_sample_data(start, periods=fallback_periods_from_start(start))
        log_df_details(f"raw {t}", df)
        data[t] = df
    log_offline_mode("extract_data")
    return data


def merge_exogenous_features(
    df: pd.DataFrame,
    exogenous_df: pd.DataFrame,
    *,
    max_ffill_steps: int | None = 3,
) -> pd.DataFrame:
    """Merge exogenous series by time using left join and controlled ffill."""
    if df.empty or exogenous_df.empty:
        return df

    base = df.copy()
    base.index = pd.to_datetime(base.index)
    if getattr(base.index, "tz", None) is not None:
        base.index = base.index.tz_convert(None)
    base = base.sort_index()

    exog = exogenous_df.copy()
    exog.index = pd.to_datetime(exog.index)
    if getattr(exog.index, "tz", None) is not None:
        exog.index = exog.index.tz_convert(None)
    exog = exog[~exog.index.duplicated(keep="last")].sort_index()

    merged = base.join(exog, how="left")
    exog_cols = [c for c in exog.columns if c in merged.columns]
    if exog_cols:
        merged[exog_cols] = merged[exog_cols].ffill(limit=max_ffill_steps)

    return merged


def enrich_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Add common technical indicators and optional exogenous features."""
    if df.empty:
        log_df_details("enrich empty", df)
        return df
    from .features import add_technical_indicators

    df = add_technical_indicators(df)
    cfg = CONFIG if config is None else config
    exog = load_exogenous_factors(
        cfg,
        index=df.index,
        start=str(df.index.min().date()),
        end=str(df.index.max().date()),
    )
    exog_cfg = cfg.get("exogenous", {}) if isinstance(cfg, dict) else {}
    max_ffill_steps = exog_cfg.get("max_ffill_steps", 3)
    df = merge_exogenous_features(df, exog, max_ffill_steps=max_ffill_steps)

    log_df_details("enriched", df)
    return df


def preprocess_data(data: Dict[str, pd.DataFrame], config: dict | None = None) -> Dict[str, pd.DataFrame]:
    """Apply feature enrichment to all dataframes."""
    processed = {}
    for t, df in data.items():
        with timed_stage(f"preprocess {t}"):
            processed_df = enrich_features(df, config=config)
        log_df_details(f"processed {t}", processed_df)
        processed[t] = processed_df
    log_offline_mode("preprocess_data")
    return processed
