"""Portfolio optimization utilities"""
import logging
from pathlib import Path
from typing import Dict
import pandas as pd


def optimize_portfolio(abt_path: Path, config: Dict) -> Path:
    """Simple equal-weight optimizer."""
    logger = logging.getLogger(__name__)
    df = pd.read_csv(abt_path)
    assets = df.columns.tolist()
    weights = {a: 1 / len(assets) for a in assets}
    logger.debug("Weights: %s", weights)

    data_dir = Path(config.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    weights_path = data_dir / "weights.csv"
    pd.DataFrame([weights]).to_csv(weights_path, index=False)
    logger.info("Weights saved to %s", weights_path)
    return weights_path
