"""Backtesting utilities"""
import logging
from pathlib import Path
from typing import Dict
import pandas as pd


def run_backtest(weights_path: Path, config: Dict) -> None:
    """Placeholder backtest that logs the weights."""
    logger = logging.getLogger(__name__)
    df = pd.read_csv(weights_path)
    logger.info("Running backtest with weights:\n%s", df)
