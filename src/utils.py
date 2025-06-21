import logging
import time
from contextlib import contextmanager
from typing import Optional

import pandas as pd

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
