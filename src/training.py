"""Model training pipeline."""
import logging
from .utils import load_config
from .utils.schema_guard import hash_schema, save_with_schema
from pathlib import Path
from typing import Dict, Union, Iterable
import pandas as pd
import json

try:  # optional dependency
    from .models.lstm_model import train_lstm, predict_lstm
except Exception:  # pragma: no cover - optional tensorflow/keras
    train_lstm = predict_lstm = None
from .models.rf_model import train_rf
from .models.xgb_model import train_xgb
from .models.linear_model import train_linear
from .models.lightgbm_model import train_lgbm
from .models.arima_model import train_arima
from .utils import (
    timed_stage,
    log_df_details,
    log_offline_mode,
    rolling_cv,
    to_price,
)
from .variable_selection import select_features_rf_cv
from .evaluation import evaluate_predictions
from .permutation_importance import compute_permutation_importance

# Maximum days required by moving averages or lag features
LOOKBACK_MONTHS = 12
MAX_MA_DAYS = 50
VAL_WEEKS = 8
CV_HORIZON_DAYS = 20

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
FEATURE_DIR = Path(__file__).resolve().parents[1] / CONFIG.get(
    "feature_dir", "results/features"
)
FEATURE_DIR.mkdir(exist_ok=True, parents=True)
VIZ_DIR = Path(__file__).resolve().parents[1] / "results" / "viz"
VIZ_DIR.mkdir(exist_ok=True, parents=True)


def split_train_test(df: pd.DataFrame, val_weeks: int = VAL_WEEKS):
    """Return train and test splits ensuring no overlap."""
    if df.empty:
        return df, df

    val_start = df.index.max() - pd.Timedelta(weeks=val_weeks)
    df_train = df[df.index < val_start]
    df_test = df[df.index >= val_start]

    if not df_train.empty and not df_test.empty:
        latest_train = df_train.index.max()
        earliest_test = df_test.index.min()
        if latest_train >= earliest_test:
            raise ValueError(
                f"Train end {latest_train} overlaps test start {earliest_test}"
            )
    return df_train, df_test


def _retrain_with_perm_importance(
    importance_model,
    *,
    model_label: str,
    train_fn,
    predict_fn,
    save_fn=None,
    train_kwargs=None,
    feature_cols: Iterable[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    ticker: str,
    frequency: str,
    paths: Dict[str, Path],
    metrics_rows: list[dict],
    var_rows: list[dict],
    abt_window: str,
    train_window: str,
    test_window: str,
    predict_date,
) -> None:
    """Compute permutation importances and retrain model if needed."""
    if predict_fn is None:
        predict_fn = lambda m, X: m.predict(X)
    if save_fn is None:
        save_fn = save_with_schema
    if train_kwargs is None:
        train_kwargs = {}

    try:
        imp_df = compute_permutation_importance(
            importance_model, X_test[list(feature_cols)], y_test
        )
        used_feats = set(imp_df.loc[imp_df.importance_mean > 0, "feature"])
        for row in imp_df.itertuples(index=False):
            var_rows.append(
                {
                    "model": f"{ticker}_{model_label}",
                    "feature": row.feature,
                    "importance_mean": float(row.importance_mean),
                    "importance_std": float(row.importance_std),
                    "importance_mean_minus_std": float(row.importance_mean_minus_std),
                    "used_in_retrain": int(row.feature in used_feats),
                    "run_date": RUN_TIMESTAMP,
                }
            )

        if used_feats and used_feats != set(feature_cols):
            X_train_f = X_train[list(used_feats)]
            X_test_f = X_test[list(used_feats)]
            base_train = df_train.loc[X_train_f.index, target_col]
            base_test = df_test.loc[X_test_f.index, target_col]
            with timed_stage(f"retrain {model_label.upper()} {ticker}"):
                model = train_fn(X_train_f, y_train, **train_kwargs)
            schema_hash = hash_schema(X_train_f)
            model_path = MODEL_DIR / f"{ticker}_{frequency}_{model_label}_{schema_hash}.joblib"
            save_fn(model, model_path, list(used_feats), schema_hash)
            paths[f"{ticker}_{model_label}"] = model_path
            try:
                preds_train = predict_fn(model, X_train_f)
                train_pred_price = to_price(preds_train, base_train, "diff")
                train_true_price = to_price(y_train, base_train, "diff")
                train_metrics = evaluate_predictions(train_true_price, train_pred_price)
                metrics_rows.append(
                    {
                        "model": f"{ticker}_{model_label}",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 1,
                    }
                )
                preds_test = predict_fn(model, X_test_f)
                test_pred_price = to_price(preds_test, base_test, "diff")
                test_true_price = to_price(y_test, base_test, "diff")
                test_metrics = evaluate_predictions(test_true_price, test_pred_price)
                metrics_rows.append(
                    {
                        "model": f"{ticker}_{model_label}",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 1,
                    }
                )
            except Exception:
                logger.exception("Failed evaluation for retrained %s %s", model_label.upper(), ticker)
        else:
            for row in metrics_rows[-2:]:
                row["retrained"] = 0
    except Exception:
        logger.exception("Failed permutation importance for %s %s", model_label.upper(), ticker)



def train_models(
    data: Union[Dict[str, Union[pd.DataFrame, Path]], pd.DataFrame],
    frequency: str = "daily",
) -> Dict[str, Path]:
    """Train basic models, evaluate them and persist both models and metrics."""
    paths = {}
    metrics_rows = []
    var_rows = []
    pred_rows = []

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
        df_recent["delta"] = df_recent[target_col].diff()
        df_recent["target"] = df_recent["delta"].shift(-1)
        df_recent.dropna(inplace=True)

        X = df_recent.drop(columns=[target_col, "target"], errors="ignore")
        y = df_recent["target"]
        log_df_details(f"features {ticker}", X)

        df_train, df_test = split_train_test(df_recent)

        abt_start = df_recent.index.min().date()
        abt_end = df_recent.index.max().date()
        train_start = df_train.index.min().date()
        train_end = df_train.index.max().date()
        test_start = df_test.index.min().date()
        test_end = df_test.index.max().date()
        abt_window = f"{abt_start} a {abt_end}"
        train_window = f"{train_start} a {train_end}"
        test_window = f"{test_start} a {test_end}"
        predict_date = test_end
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
        feature_cols = list(X.columns)
        X_train = df_train[feature_cols]
        y_train = df_train["target"]
        X_test = df_test[feature_cols]
        y_test = df_test["target"]

        train_pred_df = pd.DataFrame(
            {
                "Fecha": X_train.index,
                "ticker": ticker,
                "Valor original": y_train.values,
                "Dataset": "Train",
            }
        )
        test_pred_df = pd.DataFrame(
            {
                "Fecha": X_test.index,
                "ticker": ticker,
                "Valor original": y_test.values,
                "Dataset": "Test",
            }
        )

        logger.info("%s selected features: %s", ticker, selected_cols)
        log_df_details(f"train features {ticker}", X_train)
        log_df_details(f"test features {ticker}", X_test)

        n_samples = len(df_train)
        cv_splitter = rolling_cv(n_samples, horizon=CV_HORIZON_DAYS, max_splits=24)

        with timed_stage(f"train Linear {ticker}"):
            try:
                lin = train_linear(X_train, y_train, cv=cv_splitter)
                schema_hash = hash_schema(X_train)
                lin_path = MODEL_DIR / f"{ticker}_{frequency}_linreg_{schema_hash}.joblib"
                save_with_schema(lin, lin_path, feature_cols, schema_hash)
                paths[f"{ticker}_linreg"] = lin_path
                try:
                    preds_train = lin.predict(X_train)
                    train_pred_df["LINREG"] = preds_train
                    base_train = df_train.loc[X_train.index, target_col]
                    train_pred_price = to_price(preds_train, base_train, "diff")
                    train_true_price = to_price(y_train, base_train, "diff")
                    train_metrics = evaluate_predictions(train_true_price, train_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_linreg",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    preds_test = lin.predict(X_test)
                    test_pred_df["LINREG"] = preds_test
                    base_test = df_test.loc[X_test.index, target_col]
                    test_pred_price = to_price(preds_test, base_test, "diff")
                    test_true_price = to_price(y_test, base_test, "diff")
                    test_metrics = evaluate_predictions(test_true_price, test_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_linreg",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    logger.info(
                        "Linear train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.exception("Failed Linear evaluation for %s", ticker)
            except Exception:
                logger.exception("Failed Linear training for %s", ticker)
        if 'lin' in locals():
            _retrain_with_perm_importance(
                lin,
                model_label='linreg',
                train_fn=train_linear,
                train_kwargs={'cv': cv_splitter},
                predict_fn=lambda m, X: m.predict(X),
                feature_cols=feature_cols,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                df_train=df_train,
                df_test=df_test,
                target_col=target_col,
                ticker=ticker,
                frequency=frequency,
                paths=paths,
                metrics_rows=metrics_rows,
                var_rows=var_rows,
                abt_window=abt_window,
                train_window=train_window,
                test_window=test_window,
                predict_date=predict_date,
            )

        with timed_stage(f"train RF {ticker}"):
            try:
                rf_grid = {
                    "max_depth": [3, 4],
                    "max_features": [0.5],
                    "min_samples_leaf": [2, 5],
                }
                rf = train_rf(
                    X_train,
                    y_train,
                    param_grid=rf_grid,
                    cv=cv_splitter,
                    n_estimators=100,
                )
                schema_hash = hash_schema(X_train)
                rf_path = MODEL_DIR / f"{ticker}_{frequency}_rf_{schema_hash}.joblib"
                save_with_schema(rf, rf_path, feature_cols, schema_hash)
                paths[f"{ticker}_rf"] = rf_path
                try:
                    preds_train = rf.predict(X_train)
                    train_pred_df["RF"] = preds_train
                    base_train = df_train.loc[X_train.index, target_col]
                    train_pred_price = to_price(preds_train, base_train, "diff")
                    train_true_price = to_price(y_train, base_train, "diff")
                    train_metrics = evaluate_predictions(train_true_price, train_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_rf",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    preds_test = rf.predict(X_test)
                    test_pred_df["RF"] = preds_test
                    base_test = df_test.loc[X_test.index, target_col]
                    test_pred_price = to_price(preds_test, base_test, "diff")
                    test_true_price = to_price(y_test, base_test, "diff")
                    test_metrics = evaluate_predictions(test_true_price, test_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_rf",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    logger.info(
                        "RF train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.exception("Failed RF evaluation for %s", ticker)
            except Exception:
                logger.exception("Failed RF training for %s", ticker)
        if 'rf' in locals():
            _retrain_with_perm_importance(
                rf,
                model_label='rf',
                train_fn=train_rf,
                train_kwargs={'cv': cv_splitter},
                predict_fn=lambda m, X: m.predict(X),
                feature_cols=feature_cols,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                df_train=df_train,
                df_test=df_test,
                target_col=target_col,
                ticker=ticker,
                frequency=frequency,
                paths=paths,
                metrics_rows=metrics_rows,
                var_rows=var_rows,
                abt_window=abt_window,
                train_window=train_window,
                test_window=test_window,
                predict_date=predict_date,
            )

        with timed_stage(f"train XGB {ticker}"):
            try:
                xgb_grid = {
                    "max_depth": [3, 4],
                    "min_child_weight": [5],
                    "learning_rate": [0.05],
                    "subsample": [0.8],
                    "colsample_bytree": [0.8],
                    "n_estimators": [100],
                }
                xgb = train_xgb(
                    X_train,
                    y_train,
                    param_grid=xgb_grid,
                    cv=cv_splitter,
                )
                schema_hash = hash_schema(X_train)
                xgb_path = MODEL_DIR / f"{ticker}_{frequency}_xgb_{schema_hash}.joblib"
                save_with_schema(xgb, xgb_path, feature_cols, schema_hash)
                paths[f"{ticker}_xgb"] = xgb_path
                try:
                    preds_train = xgb.predict(X_train)
                    train_pred_df["XGB"] = preds_train
                    base_train = df_train.loc[X_train.index, target_col]
                    train_pred_price = to_price(preds_train, base_train, "diff")
                    train_true_price = to_price(y_train, base_train, "diff")
                    train_metrics = evaluate_predictions(train_true_price, train_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_xgb",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    preds_test = xgb.predict(X_test)
                    test_pred_df["XGB"] = preds_test
                    base_test = df_test.loc[X_test.index, target_col]
                    test_pred_price = to_price(preds_test, base_test, "diff")
                    test_true_price = to_price(y_test, base_test, "diff")
                    test_metrics = evaluate_predictions(test_true_price, test_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_xgb",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    logger.info(
                        "XGB train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.exception("Failed XGB evaluation for %s", ticker)
            except Exception:
                logger.exception("Failed XGB training for %s", ticker)
        if 'xgb' in locals():
            _retrain_with_perm_importance(
                xgb,
                model_label='xgb',
                train_fn=train_xgb,
                train_kwargs={'cv': cv_splitter},
                predict_fn=lambda m, X: m.predict(X),
                feature_cols=feature_cols,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                df_train=df_train,
                df_test=df_test,
                target_col=target_col,
                ticker=ticker,
                frequency=frequency,
                paths=paths,
                metrics_rows=metrics_rows,
                var_rows=var_rows,
                abt_window=abt_window,
                train_window=train_window,
                test_window=test_window,
                predict_date=predict_date,
            )

        with timed_stage(f"train LGBM {ticker}"):
            try:
                lgbm_grid = {
                    "max_depth": [3, 4],
                    "learning_rate": [0.05],
                    "subsample": [0.8],
                    "colsample_bytree": [0.8],
                    "n_estimators": [100],
                }
                lgbm = train_lgbm(
                    X_train,
                    y_train,
                    param_grid=lgbm_grid,
                    cv=cv_splitter,
                )
                schema_hash = hash_schema(X_train)
                lgbm_path = MODEL_DIR / f"{ticker}_{frequency}_lgbm_{schema_hash}.joblib"
                save_with_schema(lgbm, lgbm_path, feature_cols, schema_hash)
                paths[f"{ticker}_lgbm"] = lgbm_path
                try:
                    preds_train = lgbm.predict(X_train)
                    train_pred_df["LGBM"] = preds_train
                    base_train = df_train.loc[X_train.index, target_col]
                    train_pred_price = to_price(preds_train, base_train, "diff")
                    train_true_price = to_price(y_train, base_train, "diff")
                    train_metrics = evaluate_predictions(train_true_price, train_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_lgbm",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    preds_test = lgbm.predict(X_test)
                    test_pred_df["LGBM"] = preds_test
                    base_test = df_test.loc[X_test.index, target_col]
                    test_pred_price = to_price(preds_test, base_test, "diff")
                    test_true_price = to_price(y_test, base_test, "diff")
                    test_metrics = evaluate_predictions(test_true_price, test_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_lgbm",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    logger.info(
                        "LGBM train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.exception("Failed LGBM evaluation for %s", ticker)
            except Exception:
                logger.exception("Failed LGBM training for %s", ticker)
        if 'lgbm' in locals():
            _retrain_with_perm_importance(
                lgbm,
                model_label='lgbm',
                train_fn=train_lgbm,
                train_kwargs={'cv': cv_splitter},
                predict_fn=lambda m, X: m.predict(X),
                feature_cols=feature_cols,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                df_train=df_train,
                df_test=df_test,
                target_col=target_col,
                ticker=ticker,
                frequency=frequency,
                paths=paths,
                metrics_rows=metrics_rows,
                var_rows=var_rows,
                abt_window=abt_window,
                train_window=train_window,
                test_window=test_window,
                predict_date=predict_date,
            )

        with timed_stage(f"train LSTM {ticker}"):
            try:
                # narrow search space to speed up LSTM training
                lstm_grid = {
                    "units": [16],
                    "batch": [64],
                    "epochs": [2],
                    "l2_reg": [0.001],
                }
                lstm = train_lstm(
                    X_train,
                    y_train,
                    param_space=lstm_grid,
                    n_iter=1,
                    cv_splits=cv_splitter.n_splits,
                )
                keras_path = MODEL_DIR / f"{ticker}_{frequency}_lstm.keras"
                lstm.save(keras_path)
                features_path = keras_path.with_name(keras_path.stem + '_features.json')
                with open(features_path, 'w') as fh:
                    json.dump(feature_cols, fh)
                paths[f"{ticker}_lstm"] = keras_path
                try:
                    preds_train = predict_lstm(lstm, X_train)
                    train_pred_df["LSTM"] = preds_train
                    base_train = df_train.loc[X_train.index, target_col]
                    train_pred_price = to_price(preds_train, base_train, "diff")
                    train_true_price = to_price(y_train, base_train, "diff")
                    train_metrics = evaluate_predictions(train_true_price, train_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_lstm",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    preds_test = predict_lstm(lstm, X_test)
                    test_pred_df["LSTM"] = preds_test
                    base_test = df_test.loc[X_test.index, target_col]
                    test_pred_price = to_price(preds_test, base_test, "diff")
                    test_true_price = to_price(y_test, base_test, "diff")
                    test_metrics = evaluate_predictions(test_true_price, test_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_lstm",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    logger.info(
                        "LSTM train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.exception("Failed LSTM evaluation for %s", ticker)
            except Exception:
                logger.exception("Failed LSTM training for %s", ticker)
        if 'lstm' in locals():
            def _save_lstm(model, path, feats, _hash):
                model.save(path)
                with open(path.with_name(path.stem + '_features.json'), 'w') as fh:
                    json.dump(feats, fh)

            wrapper = type('Wrapper', (), {'predict': lambda self, X: predict_lstm(lstm, X)})()
            _retrain_with_perm_importance(
                wrapper,
                model_label='lstm',
                train_fn=lambda X, y: train_lstm(
                    X,
                    y,
                    cv_splits=cv_splitter.n_splits,
                ),
                predict_fn=predict_lstm,
                save_fn=_save_lstm,
                feature_cols=feature_cols,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                df_train=df_train,
                df_test=df_test,
                target_col=target_col,
                ticker=ticker,
                frequency=frequency,
                paths=paths,
                metrics_rows=metrics_rows,
                var_rows=var_rows,
                abt_window=abt_window,
                train_window=train_window,
                test_window=test_window,
                predict_date=predict_date,
            )

        with timed_stage(f"train ARIMA {ticker}"):
            try:
                arima = train_arima(y_train)
                schema_hash = hash_schema(X_train)
                arima_path = MODEL_DIR / f"{ticker}_{frequency}_arima_{schema_hash}.joblib"
                save_with_schema(arima, arima_path, feature_cols, schema_hash)
                paths[f"{ticker}_arima"] = arima_path
                try:
                    preds_train = arima.predict(X_train)
                    train_pred_df["ARIMA"] = preds_train
                    base_train = df_train.loc[X_train.index, target_col]
                    train_pred_price = to_price(preds_train, base_train, "diff")
                    train_true_price = to_price(y_train, base_train, "diff")
                    train_metrics = evaluate_predictions(train_true_price, train_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_arima",
                        "dataset": "train",
                        **train_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    preds_test = arima.predict(X_test)
                    test_pred_df["ARIMA"] = preds_test
                    base_test = df_test.loc[X_test.index, target_col]
                    test_pred_price = to_price(preds_test, base_test, "diff")
                    test_true_price = to_price(y_test, base_test, "diff")
                    test_metrics = evaluate_predictions(test_true_price, test_pred_price)
                    metrics_rows.append({
                        "model": f"{ticker}_arima",
                        "dataset": "test",
                        **test_metrics,
                        "run_date": RUN_TIMESTAMP,
                        "ABT Window": abt_window,
                        "Train Window": train_window,
                        "Test Window": test_window,
                        "Predict Date": predict_date,
                        "retrained": 0,
                    })
                    logger.info(
                        "ARIMA train metrics %s | test metrics %s",
                        train_metrics,
                        test_metrics,
                    )
                except Exception:
                    logger.exception("Failed ARIMA evaluation for %s", ticker)
            except Exception:
                logger.exception("Failed ARIMA training for %s", ticker)
        if 'arima' in locals():
            _retrain_with_perm_importance(
                arima,
                model_label='arima',
                train_fn=lambda X, y: train_arima(y),
                predict_fn=lambda m, X: m.predict(X),
                feature_cols=feature_cols,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                df_train=df_train,
                df_test=df_test,
                target_col=target_col,
                ticker=ticker,
                frequency=frequency,
                paths=paths,
                metrics_rows=metrics_rows,
                var_rows=var_rows,
                abt_window=abt_window,
                train_window=train_window,
                test_window=test_window,
                predict_date=predict_date,
            )

        pred_rows.extend([train_pred_df, test_pred_df])

    if pred_rows:
        preds_df = pd.concat(pred_rows, ignore_index=True)
        viz_file = VIZ_DIR / "viz_prediction.csv"
        try:
            preds_df.to_csv(viz_file, index=False)
            logger.info("Saved visualization predictions to %s", viz_file)
        except Exception:
            logger.exception("Failed to save visualization predictions to %s", viz_file)

        pred_dir = Path(__file__).resolve().parents[1] / "results" / "trainingpreds"
        pred_dir.mkdir(exist_ok=True, parents=True)
        pred_file = pred_dir / "fullpredict.csv"
        try:
            preds_df.to_csv(pred_file, index=False)
            logger.info("Saved training predictions to %s", pred_file)
        except Exception:
            logger.exception("Failed to save training predictions to %s", pred_file)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_file = EVAL_DIR / f"metrics_{frequency}_{RUN_TIMESTAMP[:10]}.csv"
        try:
            metrics_df.to_csv(metrics_file, index=False)
            logger.info("Saved evaluation metrics to %s", metrics_file)
        except Exception:
            logger.exception("Failed to save metrics to %s", metrics_file)
        logger.info("Metrics summary:\n%s", metrics_df)

    if var_rows:
        var_df = pd.DataFrame(var_rows)
        var_file = FEATURE_DIR / f"features_{frequency}_{RUN_TIMESTAMP[:10]}.csv"
        try:
            var_df.to_csv(var_file, index=False)
            logger.info("Saved variable importances to %s", var_file)
        except Exception:
            logger.exception("Failed to save variable importances to %s", var_file)

    log_offline_mode("training")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    from .abt.build_abt import build_abt

    import argparse

    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument(
        "--frequency",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="data frequency to use",
    )
    args = parser.parse_args()

    data_paths = build_abt(args.frequency)
    combined_path = data_paths.get("combined")
    if combined_path:
        combined_df = pd.read_csv(combined_path, index_col=0, parse_dates=True)
        train_models(combined_df, frequency=args.frequency)
    else:
        data = {
            t: pd.read_csv(p, index_col=0, parse_dates=True)
            for t, p in data_paths.items()
        }
        train_models(data, frequency=args.frequency)
