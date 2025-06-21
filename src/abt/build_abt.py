"""Build ABT with dummy data and technical indicators"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


def build_abt(config: Dict) -> Path:
    """Generate a simple ABT CSV using random data."""
    logger = logging.getLogger(__name__)
    logger.info("Starting ABT generation")

    etfs = config.get("etfs", [])
    start_date = pd.to_datetime(config.get("start_date", "2015-01-01"))
    periods = config.get("prediction_horizon", 5) * 2

    index = pd.date_range(start_date, periods=periods, freq="D")
    data = {etf: np.random.rand(len(index)) for etf in etfs}
    df = pd.DataFrame(data, index=index)
    logger.debug("ABT head:\n%s", df.head())

    data_dir = Path(config.get("data_dir", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    abt_path = data_dir / "abt.csv"
    df.to_csv(abt_path)
    logger.info("ABT saved to %s", abt_path)

    return abt_path


if __name__ == "__main__":
    import yaml, os

    cfg_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    build_abt(cfg)
