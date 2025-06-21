"""Backtest utilities for strategies."""
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def run_backtest(strategy: Any, data) -> Any:
    start = time.perf_counter()
    logger.info("Starting backtest")
    try:
        # Placeholder for backtesting logic
        results = None
    except Exception:
        logger.exception("Error during backtest")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Backtest finished in %.2f seconds", duration)
    return results
