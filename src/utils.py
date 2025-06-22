import logging
import time
from contextlib import contextmanager
from typing import Optional

from sklearn.model_selection import TimeSeriesSplit

import numpy as np
import pandas as pd

# Flag to track whether sample data was generated at any stage
SAMPLE_DATA_USED = False

@contextmanager
def timed_stage(name: str):
    """Context manager to log start/end time of a stage."""
    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    logger.info("Starting %s", name)
    try:
        yield
    except Exception:
        logger.exception("Exception in %s", name)
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Finished %s in %.2f seconds", name, duration)


def log_df_details(name: str, df: Optional[pd.DataFrame], head: int = 5) -> None:
    """Log basic DataFrame information."""
    logger = logging.getLogger(__name__)
    if df is None:
        logger.info("%s: DataFrame is None", name)
        return
    rows, cols = df.shape
    logger.info("%s shape: %d rows, %d columns", name, rows, cols)
    if not df.empty:
        preview = df.head(head).to_string()
        logger.info("%s head:\n%s", name, preview)


def generate_sample_data(start: str, periods: int = 30) -> pd.DataFrame:
    """Return a simple deterministic OHLCV DataFrame for offline use."""
    global SAMPLE_DATA_USED
    SAMPLE_DATA_USED = True
    dates = pd.date_range(start=start, periods=periods, freq="D")
    base = np.linspace(1, periods, periods)
    df = pd.DataFrame(
        {
            "Open": base,
            "High": base + 1,
            "Low": base - 1,
            "Close": base,
            "Adj Close": base,
            "Volume": np.random.randint(1000, 10000, size=periods),
        },
        index=dates,
    )
    return df


def rolling_cv(
    n_samples: int,
    train_size: int = 60,
    horizon: int = 1,
    max_splits: int = 5,
) -> TimeSeriesSplit:
    """Return a rolling TimeSeriesSplit for short-term forecasting."""
    n_splits = min(max_splits, max(1, n_samples - train_size))
    return TimeSeriesSplit(
        n_splits=n_splits,
        test_size=horizon,
        max_train_size=train_size,
    )


def log_offline_mode(stage: str) -> None:
    """If sample data was used, log this fact for the given stage."""
    logger = logging.getLogger(__name__)
    if SAMPLE_DATA_USED:
        logger.info("Using generated sample data in %s stage", stage)
