"""LSTM model utilities with logging"""
import logging
from pathlib import Path
from typing import Dict
import pandas as pd


def train_lstm_model(abt_path: Path, config: Dict) -> Path:
    """Dummy trainer that records the ABT shape and saves a placeholder model."""
    logger.info("Training LSTM model")
    df = pd.read_csv(abt_path)
    logger.debug("Loaded ABT with shape %s", df.shape)

    model_dir = Path(config.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lstm_model.h5"
    df.head(1).to_csv(model_path)
    logger.info("Model saved to %s", model_path)
    return model_path


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    train_lstm_model(Path(cfg.get("data_dir", "data")) / "abt.csv", cfg)
