"""Model training pipeline."""
import logging
from .utils import load_config
from pathlib import Path
from typing import Dict, Union, Iterable

import joblib
import pandas as pd

from .models.lstm_model import train_lstm
from .models.rf_model import train_rf
from .models.xgb_model import train_xgb
from .models.linear_model import train_linear
from .models.lightgbm_model import train_lgbm
from .models.arima_model import train_arima
from .utils import timed_stage, log_df_details, log_offline_mode, rolling_cv
from .variable_selection import select_features_rf_cv
from .evaluation import evaluate_predictions

# Maximum days required by moving averages or lag features
LOOKBACK_MONTHS = 12
MAX_MA_DAYS = 50

RUN_TIMESTAMP = pd.Timestamp.now(tz="UTC").isoformat()

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
CONFIG = load_config(CONFIG_PATH)

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
        if target_col not in df.columns:
            logger.warning(
                "%s missing column %s, falling back to 'Close'", ticker, target_col
            )
            target_col = "Close"

        end_dt = df.index.max()
        start_cv = end_dt - pd.DateOffset(months=LOOKBACK_MONTHS)
        feature_start = start_cv - pd.Timedelta(days=MAX_MA_DAYS)
        df_recent = df.loc[df.index >= feature_start]

        df_recent = df_recent.copy()
        df_recent["target"] = df_recent[target_col].shift(-1)
        df_recent.dropna(inplace=True)

        X = df_recent.drop(columns=[target_col, "target"], errors="ignore")
        y = df_recent["target"]
        log_df_details(f"features {ticker}", X)

        val_start = df_recent.index.max() - pd.Timedelta(days=7)
        df_train = df_recent[df_recent.index < val_start]
        df_test = df_recent[df_recent.index >= val_start]
        logger.info(
            "%s train %s to %s | validation %s to %s",
            ticker,
            df_train.index.min().date(),
            df_train.index.max().date(),
            df_test.index.min().date() if not df_test.empty else None,
            df_test.index.max().date() if not df_test.empty else None,
        )

        if len(df_train) < 20:
            logger.warning(
                "%s only %d rows after cleaning; skipping training", ticker, len(df_train)
            )
            continue
        if df_test.empty:
            logger.warning("%s has no test data; skipping training", ticker)
            continue

        fs_df = df_train.drop(columns=[target_col], errors="ignore")
        selected_cols = select_features_rf_cv(fs_df, target_col="target")
        X_train = df_train[selected_cols]
        y_train = df_train["target"]
        X_test = df_test[selected_cols]
        y_test = df_test["target"]
        logger.info("%s selected features: %s", ticker, selected_cols)
        log_df_details(f"train features {ticker}", X_train)
        log_df_details(f"test features {ticker}", X_test)

        n_samples = len(df_train)
        cv_splitter = rolling_cv(n_samples)

        with timed_stage(f"train Linear {ticker}"):
            try:
                lin = train_linear(X_train, y_train, cv=cv_splitter)
                lin_path = MODEL_DIR / f"{ticker}_{frequency}_linreg.pkl"
                joblib.dump(lin, lin_path)
                paths[f"{ticker}_linreg"] = lin_path
                try:
                    preds_train = lin.predict(X_train)
                    train_metrics = evaluate_predictions(y_train, preds_train)
                    metrics_rows.append({
                        "model": f"{ticker}_linreg",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    preds_test = lin.predict(X_test)
                    test_metrics = evaluate_predictions(y_test, preds_test)
                    metrics_rows.append({
                        "model": f"{ticker}_linreg",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    logger.info(
                        "Linear train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.error("Failed Linear evaluation for %s", ticker)
            except Exception:
                logger.error("Failed Linear training for %s", ticker)

        with timed_stage(f"train RF {ticker}"):
            try:
                rf_grid = {
                    "n_estimators": [20, 30],
                    "max_depth": [3, None],
                    "min_samples_leaf": [1, 5],
                }
                rf = train_rf(X_train, y_train, param_grid=rf_grid, cv=cv_splitter)
                rf_path = MODEL_DIR / f"{ticker}_{frequency}_rf.pkl"
                joblib.dump(rf, rf_path)
                paths[f"{ticker}_rf"] = rf_path
                try:
                    preds_train = rf.predict(X_train)
                    train_metrics = evaluate_predictions(y_train, preds_train)
                    metrics_rows.append({
                        "model": f"{ticker}_rf",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    preds_test = rf.predict(X_test)
                    test_metrics = evaluate_predictions(y_test, preds_test)
                    metrics_rows.append({
                        "model": f"{ticker}_rf",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    logger.info(
                        "RF train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.error("Failed RF evaluation for %s", ticker)
            except Exception:
                logger.error("Failed RF training for %s", ticker)

        with timed_stage(f"train XGB {ticker}"):
            try:
                xgb_grid = {
                    "n_estimators": [50, 75],
                    "max_depth": [3, 4],
                    "learning_rate": [0.05, 0.1],
                }
                xgb = train_xgb(X_train, y_train, param_grid=xgb_grid, cv=cv_splitter)
                xgb_path = MODEL_DIR / f"{ticker}_{frequency}_xgb.pkl"
                joblib.dump(xgb, xgb_path)
                paths[f"{ticker}_xgb"] = xgb_path
                try:
                    preds_train = xgb.predict(X_train)
                    train_metrics = evaluate_predictions(y_train, preds_train)
                    metrics_rows.append({
                        "model": f"{ticker}_xgb",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    preds_test = xgb.predict(X_test)
                    test_metrics = evaluate_predictions(y_test, preds_test)
                    metrics_rows.append({
                        "model": f"{ticker}_xgb",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    logger.info(
                        "XGB train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.error("Failed XGB evaluation for %s", ticker)
            except Exception:
                logger.error("Failed XGB training for %s", ticker)

        with timed_stage(f"train LGBM {ticker}"):
            try:
                lgbm_grid = {
                    "n_estimators": [50, 75],
                    "max_depth": [3, 4],
                    "learning_rate": [0.05, 0.1],
                }
                lgbm = train_lgbm(X_train, y_train, param_grid=lgbm_grid, cv=cv_splitter)
                lgbm_path = MODEL_DIR / f"{ticker}_{frequency}_lgbm.pkl"
                joblib.dump(lgbm, lgbm_path)
                paths[f"{ticker}_lgbm"] = lgbm_path
                try:
                    preds_train = lgbm.predict(X_train)
                    train_metrics = evaluate_predictions(y_train, preds_train)
                    metrics_rows.append({
                        "model": f"{ticker}_lgbm",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    preds_test = lgbm.predict(X_test)
                    test_metrics = evaluate_predictions(y_test, preds_test)
                    metrics_rows.append({
                        "model": f"{ticker}_lgbm",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    logger.info(
                        "LGBM train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.error("Failed LGBM evaluation for %s", ticker)
            except Exception:
                logger.error("Failed LGBM training for %s", ticker)

        with timed_stage(f"train LSTM {ticker}"):
            try:
                lstm_grid = {
                    "units": [16, 32],
                    "epochs": [2, 3],
                    "dropout": [0.0, 0.2],
                    "l2_reg": [0.0, 0.001],
                }
                lstm = train_lstm(X_train, y_train, param_grid=lstm_grid, cv=cv_splitter)
                lstm_path = MODEL_DIR / f"{ticker}_{frequency}_lstm.pkl"
                lstm.save(lstm_path.with_suffix('.keras'))
                paths[f"{ticker}_lstm"] = lstm_path.with_suffix('.keras')
                try:
                    preds_train = lstm.predict(X_train)
                    preds_train = preds_train.flatten() if hasattr(preds_train, "flatten") else preds_train
                    train_metrics = evaluate_predictions(y_train, preds_train)
                    metrics_rows.append({
                        "model": f"{ticker}_lstm",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    preds_test = lstm.predict(X_test)
                    preds_test = preds_test.flatten() if hasattr(preds_test, "flatten") else preds_test
                    test_metrics = evaluate_predictions(y_test, preds_test)
                    metrics_rows.append({
                        "model": f"{ticker}_lstm",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    logger.info(
                        "LSTM train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.error("Failed LSTM evaluation for %s", ticker)
            except Exception:
                logger.error("Failed LSTM training for %s", ticker)

        with timed_stage(f"train ARIMA {ticker}"):
            try:
                arima = train_arima(y_train)
                arima_path = MODEL_DIR / f"{ticker}_{frequency}_arima.pkl"
                joblib.dump(arima, arima_path)
                paths[f"{ticker}_arima"] = arima_path
                try:
                    preds_train = arima.predict(X_train)
                    train_metrics = evaluate_predictions(y_train, preds_train)
                    metrics_rows.append({
                        "model": f"{ticker}_arima",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    preds_test = arima.predict(X_test)
                    test_metrics = evaluate_predictions(y_test, preds_test)
                    metrics_rows.append({
                        "model": f"{ticker}_arima",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                    })
                    logger.info(
                        "ARIMA train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.error("Failed ARIMA evaluation for %s", ticker)
            except Exception:
                logger.error("Failed ARIMA training for %s", ticker)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_file = EVAL_DIR / f"metrics_{frequency}_{RUN_TIMESTAMP[:10]}.csv"
        try:
            metrics_df.to_csv(metrics_file, index=False)
            logger.info("Saved evaluation metrics to %s", metrics_file)
        except Exception:
            logger.error("Failed to save metrics to %s", metrics_file)
        logger.info("Metrics summary:\n%s", metrics_df)

    log_offline_mode("training")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
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
