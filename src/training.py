"""Model training pipeline."""
import logging
import yaml
from pathlib import Path
from typing import Dict, Union, Iterable

import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .models.lstm_model import train_lstm
from .models.rf_model import train_rf
from .models.xgb_model import train_xgb
from .models.linear_model import train_linear
from .models.lightgbm_model import train_lgbm
from .models.arima_model import train_arima
from .utils import timed_stage, log_df_details, log_offline_mode
from .evaluation import evaluate_predictions

RUN_TIMESTAMP = pd.Timestamp.now(tz="UTC").isoformat()

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH) as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

logger = logging.getLogger(__name__)

TARGET_COLS = CONFIG.get("target_cols", {})


MODEL_DIR = Path(__file__).resolve().parents[1] / CONFIG.get("model_dir", "models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
EVAL_DIR = Path(__file__).resolve().parents[1] / CONFIG.get(
    "evaluation_dir", "results/metrics"
)
EVAL_DIR.mkdir(exist_ok=True, parents=True)


def train_models(
    data: Union[Dict[str, Union[pd.DataFrame, Path]], pd.DataFrame],
    frequency: str = "daily",
) -> Dict[str, Path]:
    """Train basic models, evaluate them and persist both models and metrics."""
    paths = {}
    metrics_rows = []

    if isinstance(data, pd.DataFrame):
        grouped = {
            t: df.drop(columns=["Ticker"], errors="ignore")
            for t, df in data.groupby("Ticker")
        }
    else:
        grouped = {}
        for t, df in data.items():
            if isinstance(df, (str, Path)):
                df = pd.read_csv(df, index_col=0, parse_dates=True)
            grouped[t] = df

    for ticker, df in grouped.items():
        log_df_details(f"training data {ticker}", df)
        target_col = TARGET_COLS.get(ticker, "Close")

        end_dt = df.index.max()
        start_cv = end_dt - pd.DateOffset(months=6)
        df_recent = df.loc[df.index >= start_cv]

        X = df_recent.drop(columns=[target_col], errors="ignore")
        y = df_recent.get(target_col)
        log_df_details(f"features {ticker}", X)

        n_samples = len(df_recent)
        if n_samples <= 61:
            logger.warning("%s insufficient data for custom CV", ticker)
            cv_splitter = TimeSeriesSplit(n_splits=2, test_size=1, max_train_size=60)
        else:
            n_splits = min(5, n_samples - 60)
            cv_splitter = TimeSeriesSplit(n_splits=n_splits, test_size=1, max_train_size=60)

        with timed_stage(f"train Linear {ticker}"):
            try:
                lin = train_linear(X, y, cv=cv_splitter)
                lin_path = MODEL_DIR / f"{ticker}_{frequency}_linreg.pkl"
                joblib.dump(lin, lin_path)
                paths[f"{ticker}_linreg"] = lin_path
                try:
                    preds = lin.predict(X)
                    metrics = evaluate_predictions(y, preds)
                    metrics_row = {"model": f"{ticker}_linreg", **metrics, "run_date": RUN_TIMESTAMP}
                    metrics_rows.append(metrics_row)
                    logger.info("Linear metrics %s", metrics_row)
                except Exception:
                    logger.error("Failed Linear evaluation for %s", ticker)
            except Exception:
                logger.error("Failed Linear training for %s", ticker)

        with timed_stage(f"train RF {ticker}"):
            try:
                rf_grid = {
                    "n_estimators": [20, 30],
                    "max_depth": [3, None],
                }
                rf = train_rf(X, y, param_grid=rf_grid, cv=cv_splitter)
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
                xgb = train_xgb(X, y, param_grid=xgb_grid, cv=cv_splitter)
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

        with timed_stage(f"train LGBM {ticker}"):
            try:
                lgbm_grid = {
                    "n_estimators": [50, 75],
                    "max_depth": [3, 4],
                }
                lgbm = train_lgbm(X, y, param_grid=lgbm_grid, cv=cv_splitter)
                lgbm_path = MODEL_DIR / f"{ticker}_{frequency}_lgbm.pkl"
                joblib.dump(lgbm, lgbm_path)
                paths[f"{ticker}_lgbm"] = lgbm_path
                try:
                    preds = lgbm.predict(X)
                    metrics = evaluate_predictions(y, preds)
                    metrics_row = {"model": f"{ticker}_lgbm", **metrics, "run_date": RUN_TIMESTAMP}
                    metrics_rows.append(metrics_row)
                    logger.info("LGBM metrics %s", metrics_row)
                except Exception:
                    logger.error("Failed LGBM evaluation for %s", ticker)
            except Exception:
                logger.error("Failed LGBM training for %s", ticker)

        with timed_stage(f"train LSTM {ticker}"):
            try:
                lstm_grid = {"units": [16, 32], "epochs": [2, 3]}
                lstm = train_lstm(X, y, param_grid=lstm_grid, cv=cv_splitter)
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

        with timed_stage(f"train ARIMA {ticker}"):
            try:
                arima = train_arima(y)
                arima_path = MODEL_DIR / f"{ticker}_{frequency}_arima.pkl"
                joblib.dump(arima, arima_path)
                paths[f"{ticker}_arima"] = arima_path
                try:
                    preds = arima.predict(X)
                    metrics = evaluate_predictions(y, preds)
                    metrics_row = {"model": f"{ticker}_arima", **metrics, "run_date": RUN_TIMESTAMP}
                    metrics_rows.append(metrics_row)
                    logger.info("ARIMA metrics %s", metrics_row)
                except Exception:
                    logger.error("Failed ARIMA evaluation for %s", ticker)
            except Exception:
                logger.error("Failed ARIMA training for %s", ticker)

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
    combined_path = data_paths.get("combined")
    if combined_path:
        combined_df = pd.read_csv(combined_path, index_col=0, parse_dates=True)
        train_models(combined_df)
    else:
        data = {
            t: pd.read_csv(p, index_col=0, parse_dates=True)
            for t, p in data_paths.items()
        }
        train_models(data)
