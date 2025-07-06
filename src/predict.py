"""Apply trained models to new data and store predictions."""
import logging
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json

import joblib
import pandas as pd

from .utils.schema_guard import load_with_schema, validate_schema

try:
    from tensorflow import keras
except Exception:  # pragma: no cover - optional dependency
    keras = None

from .utils import log_df_details, log_offline_mode, load_config, to_price

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
CONFIG = load_config(CONFIG_PATH)

TARGET_COLS = CONFIG.get("target_cols", {})

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path(__file__).resolve().parents[1] / CONFIG.get("model_dir", "models")


def _is_lfs_pointer(path: Path) -> bool:
    """Return True if the file is a Git LFS pointer."""
    try:
        with path.open("rb") as fh:
            header = fh.read(1024)
        return b"git-lfs" in header
    except Exception:
        return False


class _NaiveModel:
    """Fallback model returning the last close value or zero."""

    def predict(self, X: pd.DataFrame):
        if len(X) == 0:
            return []
        if isinstance(X, pd.DataFrame) and "Close" in X.columns:
            last = X["Close"].iloc[-1]
            return [last] * len(X)
        return [0] * len(X)


def _is_valid_model(obj: Any) -> bool:
    """Return True if the loaded object can make predictions."""
    return hasattr(obj, "predict")


def load_models(model_dir: Path) -> Dict[str, Any]:
    models = {}
    if not model_dir.exists():
        logger.warning("Model directory %s does not exist, using naive defaults", model_dir)
        return {f"{t}_naive": _NaiveModel() for t in CONFIG.get("etfs", [])}

    # keep only the most recent version of each model to avoid stale schemas
    latest: Dict[str, Path] = {}
    hash_re = re.compile(r"_[0-9a-f]{10}$")
    for file in model_dir.iterdir():
        if not file.is_file():
            continue
        if file.suffix not in {".pkl", ".joblib", ".keras"}:
            continue
        if file.stem.endswith("_features"):
            continue
        base = hash_re.sub("", file.stem)
        prev = latest.get(base)
        if not prev or file.stat().st_mtime > prev.stat().st_mtime:
            latest[base] = file

    for file in latest.values():
        if _is_lfs_pointer(file):
            logger.error(
                "%s appears to be a Git LFS pointer. Run 'git lfs pull' to fetch the model",
                file,
            )
            continue
        if file.suffix == ".pkl":
            try:
                loaded = joblib.load(file)
                if _is_valid_model(loaded):
                    models[file.stem] = loaded
                else:
                    logger.error("%s does not appear to be a trained model", file)
            except Exception:
                logger.exception("Failed to load model %s", file)
        elif file.suffix == ".joblib":
            try:
                model, feats, schema = load_with_schema(file)
                if _is_valid_model(model):
                    models[file.stem] = (model, feats, schema)
                else:
                    logger.error("%s does not appear to be a trained model", file)
            except Exception:
                logger.exception("Failed to load model %s", file)
        elif file.suffix == ".keras":
            if keras is None:
                logger.error("TensorFlow unavailable; cannot load %s", file)
                continue
            try:
                loaded = keras.models.load_model(file)
                if _is_valid_model(loaded):
                    models[file.stem] = loaded
                else:
                    logger.error("%s does not appear to be a trained model", file)
            except Exception:
                logger.exception("Failed to load model %s", file)

    if not models:
        logger.warning("No trained models found in %s, using naive defaults", model_dir)
        for ticker in CONFIG.get("etfs", []):
            models[f"{ticker}_naive"] = _NaiveModel()

    return models


def make_prediction(
    model: Any,
    last_close: float,
    X_inputs: pd.DataFrame,
    model_meta: Dict[str, Any] | None = None,
) -> float:
    """Return next-day price prediction for a single model."""
    if model_meta is None:
        model_meta = {}
    raw_pred = model.predict(X_inputs)[0]
    target_type = model_meta.get("target_type", "price")
    return to_price(raw_pred, last_close, target_type)


def run_predictions(
    models: Dict[str, Any],
    data: Dict[str, pd.DataFrame],
    frequency: str = "daily",
) -> pd.DataFrame:
    """Run predictions for the given data."""
    rows = []
    for name, model_info in models.items():
        if isinstance(model_info, tuple):
            model, feature_list, schema_hash = model_info
        else:
            model = model_info
            feature_list = None
            schema_hash = None
        ticker = name.split("_")[0]
        target_col = TARGET_COLS.get(ticker, "Close")
        df = data.get(ticker)
        if df is None or len(df) < 2:
            logger.warning("Not enough data to predict %s", ticker)
            continue
        if target_col not in df.columns:
            logger.warning(
                "%s missing column %s, falling back to 'Close'", ticker, target_col
            )
            target_col = "Close"
        df = df.copy()
        if "delta" not in df.columns:
            df["delta"] = df[target_col].diff()
        logger.info("Using target column %s for %s", target_col, ticker)
        log_df_details(f"predict data {ticker}", df)
        X = df.drop(columns=[target_col], errors="ignore")
        if "Ticker" in X:
            X = X.drop(columns=["Ticker"])
        X = X.select_dtypes(exclude="object")
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        if X.empty:
            logger.warning("All rows have NaN values for %s", ticker)
            continue
        if feature_list is not None:
            try:
                validate_schema(feature_list, X, schema_hash)
            except SystemExit:
                logger.exception("Schema mismatch for %s", name)
                continue
            X = X.reindex(columns=feature_list, fill_value=0)
        else:
            feature_file = MODEL_DIR / f"{name}_features.json"
            if feature_file.exists():
                try:
                    with feature_file.open() as fh:
                        selected = json.load(fh)
                    X = X.reindex(columns=selected, fill_value=0)
                except Exception:
                    logger.exception("Failed to align features for %s", name)
        y = df.loc[X.index, target_col]
        train_start = df.index.min().date()
        train_end = df.index.max().date()
        predict_dt = df.index.max().date()
        try:
            last_close = df[target_col].iloc[-1]
            X_last = X.tail(1)
            if keras is not None and isinstance(model, keras.Model):
                arr = np.asarray(X_last, dtype=np.float32)
                if len(getattr(model, "input_shape", [])) == 3 and arr.ndim == 2:
                    arr = arr[:, None, :]
                pred_price = make_prediction(
                    model,
                    last_close,
                    arr,
                    {"target_type": "diff"},
                )
            else:
                pred_price = make_prediction(
                    model,
                    last_close,
                    X_last,
                    {"target_type": "diff"},
                )
            
            model_name = getattr(model, "__class__", type(model)).__name__
            params = {}
            if hasattr(model, "get_params"):
                try:
                    params = model.get_params()
                except Exception:
                    logger.warning("Could not retrieve parameters for %s", name)

            rows.append({
                "ticker": ticker,
                "model": model_name,
                "parameters": params,
                "actual": last_close,
                "pred": pred_price,
                "Training Window": f"{train_start} a {train_end}",
                "Predict moment": str(predict_dt),
            })
        except Exception:
            logger.exception("Prediction failed for %s", name)
    result_df = pd.DataFrame(rows)
    pred_dir = RESULTS_DIR / "predicts"
    pred_dir.mkdir(exist_ok=True, parents=True)
    suffix = {
        "daily": "daily_predictions.csv",
        "weekly": "weekly_predictions.csv",
        "monthly": "monthly_predictions.csv",
    }.get(frequency, "predictions.csv")
    out_file = pred_dir / suffix
    try:
        result_df.to_csv(out_file, index=False)
        logger.info("Saved predictions to %s", out_file)
    except Exception:
        logger.exception("Failed to save predictions to %s", out_file)
    log_offline_mode("prediction")
    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    from .abt.build_abt import build_abt

    import argparse

    parser = argparse.ArgumentParser(description="Run predictions")
    parser.add_argument(
        "--frequency",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="data frequency to use",
    )
    args = parser.parse_args()

    data_paths = build_abt(args.frequency)
    data = {t: pd.read_csv(p, index_col=0, parse_dates=True) for t, p in data_paths.items()}
    models = load_models(MODEL_DIR)
    run_predictions(models, data, frequency=args.frequency)
