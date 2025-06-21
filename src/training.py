"""Model training pipeline."""
import logging
import yaml
from pathlib import Path
from typing import Dict, Union

import joblib
import pandas as pd

from .models.lstm_model import train_lstm
from .models.rf_model import train_rf
from .models.xgb_model import train_xgb
from .utils import timed_stage

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH) as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

logger = logging.getLogger(__name__)


MODEL_DIR = Path(__file__).resolve().parents[1] / CONFIG.get("model_dir", "models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def train_models(data: Dict[str, Union[pd.DataFrame, Path]], frequency: str = "daily") -> Dict[str, Path]:
    """Train basic models and persist them to disk."""
    paths = {}
    for ticker, df in data.items():
        if isinstance(df, (str, Path)):
            df = pd.read_csv(df)
        X = df.drop(columns=["Close"], errors="ignore")
        y = df.get("Close")

        with timed_stage(f"train RF {ticker}"):
            try:
                rf = train_rf(X, y)
                rf_path = MODEL_DIR / f"{ticker}_{frequency}_rf.pkl"
                joblib.dump(rf, rf_path)
                paths[f"{ticker}_rf"] = rf_path
            except Exception:
                logger.error("Failed RF training for %s", ticker)

        with timed_stage(f"train XGB {ticker}"):
            try:
                xgb = train_xgb(X, y)
                xgb_path = MODEL_DIR / f"{ticker}_{frequency}_xgb.pkl"
                joblib.dump(xgb, xgb_path)
                paths[f"{ticker}_xgb"] = xgb_path
            except Exception:
                logger.error("Failed XGB training for %s", ticker)

        with timed_stage(f"train LSTM {ticker}"):
            try:
                lstm = train_lstm(X, y)
                lstm_path = MODEL_DIR / f"{ticker}_{frequency}_lstm.pkl"
                joblib.dump(lstm, lstm_path)
                paths[f"{ticker}_lstm"] = lstm_path
            except Exception:
                logger.error("Failed LSTM training for %s", ticker)

    return paths


if __name__ == "__main__":
    from .abt.build_abt import build_abt

    data_paths = build_abt()
    data = {t: pd.read_csv(p) for t, p in data_paths.items()}
    train_models(data)
