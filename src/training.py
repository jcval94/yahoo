"""Model training pipeline."""
import logging
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from .models.lstm_model import train_lstm
from .models.rf_model import train_rf
from .models.xgb_model import train_xgb
from .utils import timed_stage

logger = logging.getLogger(__name__)


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def train_models(data: Dict[str, pd.DataFrame], frequency: str = "daily") -> Dict[str, Path]:
    """Train placeholder models and persist them to disk."""
    paths = {}
    for ticker, df in data.items():
        X = df.drop(columns=["Close"], errors="ignore")
        y = df.get("Close")

        with timed_stage(f"train RF {ticker}"):
            rf = train_rf(X, y)
            rf_path = MODEL_DIR / f"{ticker}_{frequency}_rf.pkl"
            joblib.dump(rf, rf_path)
            paths[f"{ticker}_rf"] = rf_path

        with timed_stage(f"train XGB {ticker}"):
            xgb = train_xgb(X, y)
            xgb_path = MODEL_DIR / f"{ticker}_{frequency}_xgb.pkl"
            joblib.dump(xgb, xgb_path)
            paths[f"{ticker}_xgb"] = xgb_path

        with timed_stage(f"train LSTM {ticker}"):
            lstm = train_lstm(X, y)
            lstm_path = MODEL_DIR / f"{ticker}_{frequency}_lstm.pkl"
            joblib.dump(lstm, lstm_path)
            paths[f"{ticker}_lstm"] = lstm_path

    return paths
