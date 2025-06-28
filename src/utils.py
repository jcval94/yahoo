import logging
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Union

from sklearn.model_selection import TimeSeriesSplit

import numpy as np
import pandas as pd

# Flag to track whether sample data was generated at any stage
SAMPLE_DATA_USED = False


def load_config(path: Union[str, Path]) -> dict:
    """Load the YAML configuration file with a fallback parser.

    The real project depends on :mod:`yaml` which may not be available in the
    execution environment.  This helper first tries to use ``yaml.safe_load`` and
    if that fails it falls back to a very small parser that understands the
    limited structure of ``config.yaml`` used in the tests.
    """

    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception:
        pass

    config: dict[str, Union[str, float, int, list[str]]] = {}
    current_key: Optional[str] = None
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":") and not line.startswith("-"):
                current_key = line[:-1]
                config[current_key] = []
                continue
            if line.startswith("- ") and isinstance(config.get(current_key), list):
                value = line[2:].strip().strip('"').strip("'")
                config[current_key].append(value)
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # attempt numeric conversion
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
                config[key] = value

    return config

@contextmanager
def timed_stage(name: str):
    """Context manager to log start/end time of a stage."""
    logger = logging.getLogger(__name__)
    start = time.perf_counter()
    logger.info("-" * 40)
    logger.info("Starting %s", name)
    try:
        yield
    except Exception:
        logger.exception("Exception in %s", name)
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Finished %s in %.2f seconds", name, duration)
        logger.info("-" * 40)


def log_df_details(name: str, df: Optional[pd.DataFrame], head: int = 5) -> None:
    """Log basic DataFrame information."""
    logger = logging.getLogger(__name__)
    if df is None:
        logger.info("%s: DataFrame is None", name)
        return
    rows, cols = df.shape
    logger.info("%s shape: %d rows, %d columns", name, rows, cols)
    if not df.empty:
        preview = df.head(head).to_string(max_cols=None)
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
    train_size: int = 90,
    horizon: int = 3,
    max_splits: int = 8,
) -> TimeSeriesSplit:
    """Return a rolling ``TimeSeriesSplit`` for forecasting.

    Parameters
    ----------
    n_samples
        Total number of observations available.
    train_size
        Size of each training window. Defaults to ``90`` days for more
        robust training.
    horizon
        Forecast horizon (size of the test window). Defaults to ``3`` days
        to better capture short-term trends.
    max_splits
        Maximum number of CV splits. Increasing this value provides
        better validation while keeping runtime reasonable.
    """
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
