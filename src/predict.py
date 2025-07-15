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
from .evaluation import evaluate_predictions

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
CONFIG = load_config(CONFIG_PATH)
RUN_TIMESTAMP = pd.Timestamp.now(tz="UTC").isoformat()

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
    raw_pred = model.predict(X_inputs)
    try:
        raw_pred = float(np.asarray(raw_pred).reshape(-1)[0])
    except Exception:
        raw_pred = float(raw_pred[0])
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
        parts = name.split("_")
        ticker = parts[0]
        algo = parts[2] if len(parts) > 2 else getattr(model, "__class__", type(model)).__name__
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
            
            model_name = algo
            params = {}
            if hasattr(model, "get_params"):
                try:
                    params = model.get_params()
                except Exception:
                    logger.warning("Could not retrieve parameters for %s", name)

            predict_date = (df.index.max() + pd.offsets.BDay()).date()

            rows.append({
                "ticker": ticker,
                "model": model_name,
                "parameters": params,
                "actual": last_close,
                "pred": pred_price,
                "Training Window": f"{train_start} a {train_end}",
                "Predict moment": str(predict_dt),
                "Predicted": str(predict_date),
            })
        except Exception:
            logger.exception("Prediction failed for %s", name)
    result_df = pd.DataFrame(rows)
    if not result_df.empty and "parameters" in result_df.columns:
        ordered_cols = [c for c in result_df.columns if c != "parameters"] + ["parameters"]
        result_df = result_df[ordered_cols]
    pred_dir = RESULTS_DIR / "predicts"
    pred_dir.mkdir(exist_ok=True, parents=True)
    suffix = {
        "daily": "daily_predictions.csv",
        "weekly": "weekly_predictions.csv",
        "monthly": "monthly_predictions.csv",
    }.get(frequency, "predictions.csv")
    out_file = pred_dir / f"{RUN_TIMESTAMP[:10]}_{suffix}"
    try:
        result_df.to_csv(out_file, index=False)
        logger.info("Saved predictions to %s", out_file)
    except Exception:
        logger.exception("Failed to save predictions to %s", out_file)
    log_offline_mode("prediction")
    return result_df


def save_edge_predictions(result_df: pd.DataFrame) -> Path:
    """Persist edge predictions for top models and ensemble."""
    edge_dir = RESULTS_DIR / "predicts"
    edge_dir.mkdir(exist_ok=True, parents=True)

    if result_df.empty:
        edge_file = edge_dir / f"{RUN_TIMESTAMP[:10]}_edge_prediction.csv"
        pd.DataFrame().to_csv(edge_file, index=False)
        return edge_file

    metrics_dir = RESULTS_DIR / "metrics"
    files = sorted(metrics_dir.glob("metrics_*_*.csv"))
    top_models: dict[str, list[str]] = {}
    if files:
        latest = files[-1]
        try:
            metrics_df = pd.read_csv(latest)
            metrics_df = metrics_df[metrics_df["dataset"] == "test"].copy()
            metrics_df["ticker"] = metrics_df["model"].str.split("_").str[0]
            metrics_df["algo"] = metrics_df["model"].str.split("_").str[1]
            for t, grp in metrics_df.groupby("ticker"):
                best = grp.sort_values("RMSE").head(3)["algo"].tolist()
                top_models[t] = best
        except Exception:
            logger.exception("Failed to load metrics from %s", latest)

    rows = []
    for ticker, group in result_df.groupby("ticker"):
        subset = group
        if top_models.get(ticker):
            subset = subset[subset["model"].isin(top_models[ticker])]
        for _, r in subset.iterrows():
            rows.append(
                {
                    "ticker": r["ticker"],
                    "model": r["model"],
                    "pred": float(r["pred"]),
                    "Predicted": r["Predicted"],
                }
            )
        if not subset.empty:
            ensemble_pred = float(subset["pred"].astype(float).mean())
            rows.append(
                {
                    "ticker": ticker,
                    "model": "Top3Ensamble",
                    "pred": ensemble_pred,
                    "Predicted": subset["Predicted"].iloc[0],
                }
            )

    edge_df = pd.DataFrame(rows)
    edge_file = edge_dir / f"{RUN_TIMESTAMP[:10]}_edge_prediction.csv"
    edge_df.to_csv(edge_file, index=False)
    logger.info("Saved edge predictions to %s", edge_file)
    return edge_file


def evaluate_edge_predictions(data: Dict[str, pd.DataFrame], prev_file: Path) -> pd.DataFrame | None:
    """Compare edge predictions with actual values and save metrics."""
    if not prev_file.exists():
        logger.info("Previous edge prediction %s not found", prev_file)
        return None

    prev_df = pd.read_csv(prev_file)
    if prev_df.empty:
        logger.warning("Previous edge prediction file %s is empty", prev_file)
        return None

    prev_df["pred"] = prev_df["pred"].apply(lambda x: float(np.asarray(x).reshape(-1)[0]))
    predicted_ts = pd.to_datetime(prev_df["Predicted"].iloc[0])
    predicted_date = predicted_ts.date()
    predicted_ts = predicted_ts.tz_localize(None) if predicted_ts.tzinfo else predicted_ts
    rows = []
    for _, row in prev_df.iterrows():
        ticker = row["ticker"]
        model_name = row["model"]
        pred_val = float(row["pred"])
        df = data.get(ticker)
        if df is None:
            continue
        idx = df.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        if predicted_ts not in idx:
            continue
        target_col = TARGET_COLS.get(ticker, "Close")
        actual_val = df.loc[df.index[idx.get_loc(predicted_ts)], target_col]
        metrics = evaluate_predictions([actual_val], [pred_val])
        rows.append({"ticker": ticker, "model": model_name, "pred": pred_val, "Predicted": str(predicted_date), **metrics})

    if not rows:
        return None

    metrics_df = pd.DataFrame(rows)

    # Save metrics inside results/edge_metrics for backwards compatibility
    metrics_dir = RESULTS_DIR / "edge_metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)
    metrics_file = metrics_dir / f"edge_metrics_{predicted_date}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    logger.info("Saved edge metrics to %s", metrics_file)

    # Also store metrics in the main metrics folder
    metrics_main_dir = RESULTS_DIR / "metrics"
    metrics_main_dir.mkdir(exist_ok=True, parents=True)
    metrics_main_file = metrics_main_dir / f"edge_metrics_{predicted_date}.csv"
    metrics_df.to_csv(metrics_main_file, index=False)
    logger.info("Saved edge metrics to %s", metrics_main_file)
    return metrics_df


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
    result = run_predictions(models, data, frequency=args.frequency)
    edge_file = save_edge_predictions(result)
    prev_date = (pd.to_datetime(RUN_TIMESTAMP).date() - pd.Timedelta(days=1))
    prev_file = RESULTS_DIR / "predicts" / f"{prev_date}_edge_prediction.csv"
    evaluate_edge_predictions(data, prev_file)
