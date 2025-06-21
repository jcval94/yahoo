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
from .utils import timed_stage, log_df_details, log_offline_mode
from .evaluation import evaluate_predictions

RUN_TIMESTAMP = pd.Timestamp.now(tz="UTC").isoformat()

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH) as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

logger = logging.getLogger(__name__)


MODEL_DIR = Path(__file__).resolve().parents[1] / CONFIG.get("model_dir", "models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
EVAL_DIR = Path(__file__).resolve().parents[1] / CONFIG.get(
    "evaluation_dir", "results/metrics"
)
EVAL_DIR.mkdir(exist_ok=True, parents=True)


def train_models(
    data: Dict[str, Union[pd.DataFrame, Path]], frequency: str = "daily"
) -> Dict[str, Path]:
    """Train basic models, evaluate them and persist both models and metrics."""
    paths = {}
    metrics_rows = []
    for ticker, df in data.items():
        if isinstance(df, (str, Path)):
            df = pd.read_csv(df, index_col=0, parse_dates=True)
        log_df_details(f"training data {ticker}", df)
        X = df.drop(columns=["Close"], errors="ignore")
        y = df.get("Close")
        log_df_details(f"features {ticker}", X)

        with timed_stage(f"train RF {ticker}"):
            try:
                rf_grid = {
                    "n_estimators": [20, 30],
                    "max_depth": [3, None],
                }
                rf = train_rf(X, y, param_grid=rf_grid, cv=2)
                rf_path = MODEL_DIR / f"{ticker}_{frequency}_rf.pkl"
                joblib.dump(rf, rf_path)
                paths[f"{ticker}_rf"] = rf_path
                try:
                    preds = rf.predict(X)
                    metrics = evaluate_predictions(y, preds)
                    metrics_row = {"model": f"{ticker}_rf", **metrics, "run_date": RUN_TIMESTAMP}
                    metrics_rows.append(metrics_row)
                    logger.info("RF metrics %s", metrics_row)
                except Exception:
                    logger.error("Failed RF evaluation for %s", ticker)
            except Exception:
                logger.error("Failed RF training for %s", ticker)

        with timed_stage(f"train XGB {ticker}"):
            try:
                xgb_grid = {
                    "n_estimators": [50, 75],
                    "max_depth": [3, 4],
                }
                xgb = train_xgb(X, y, param_grid=xgb_grid, cv=2)
                xgb_path = MODEL_DIR / f"{ticker}_{frequency}_xgb.pkl"
                joblib.dump(xgb, xgb_path)
                paths[f"{ticker}_xgb"] = xgb_path
                try:
                    preds = xgb.predict(X)
                    metrics = evaluate_predictions(y, preds)
                    metrics_row = {"model": f"{ticker}_xgb", **metrics, "run_date": RUN_TIMESTAMP}
                    metrics_rows.append(metrics_row)
                    logger.info("XGB metrics %s", metrics_row)
                except Exception:
                    logger.error("Failed XGB evaluation for %s", ticker)
            except Exception:
                logger.error("Failed XGB training for %s", ticker)

        with timed_stage(f"train LSTM {ticker}"):
            try:
                lstm_grid = {"units": [16, 32], "epochs": [2, 3]}
                lstm = train_lstm(X, y, param_grid=lstm_grid, cv=2)
                lstm_path = MODEL_DIR / f"{ticker}_{frequency}_lstm.pkl"
                joblib.dump(lstm, lstm_path)
                paths[f"{ticker}_lstm"] = lstm_path
                try:
                    preds = lstm.predict(X)
                    preds = preds.flatten() if hasattr(preds, "flatten") else preds
                    metrics = evaluate_predictions(y, preds)
                    metrics_row = {"model": f"{ticker}_lstm", **metrics, "run_date": RUN_TIMESTAMP}
                    metrics_rows.append(metrics_row)
                    logger.info("LSTM metrics %s", metrics_row)
                except Exception:
                    logger.error("Failed LSTM evaluation for %s", ticker)
            except Exception:
                logger.error("Failed LSTM training for %s", ticker)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_file = EVAL_DIR / f"metrics_{frequency}_{RUN_TIMESTAMP[:10]}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        logger.info("Saved evaluation metrics to %s", metrics_file)
        logger.info("Metrics summary:\n%s", metrics_df)

    log_offline_mode("training")
    return paths


if __name__ == "__main__":
    from .abt.build_abt import build_abt

    data_paths = build_abt()
    data = {t: pd.read_csv(p, index_col=0, parse_dates=True) for t, p in data_paths.items()}
    train_models(data)
