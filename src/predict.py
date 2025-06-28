"""Apply trained models to new data and store predictions."""
import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from tensorflow import keras

from sklearn.metrics import mean_absolute_error, r2_score

from .utils import log_df_details, log_offline_mode

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH) as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

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


def load_models(model_dir: Path) -> Dict[str, Any]:
    models = {}
    for file in model_dir.iterdir():
        if not file.is_file():
            continue
        if _is_lfs_pointer(file):
            logger.error(
                "%s appears to be a Git LFS pointer. Run 'git lfs pull' to fetch the model",
                file,
            )
            continue
        if file.suffix == ".pkl":
            try:
                models[file.stem] = joblib.load(file)
            except Exception:
                logger.error("Failed to load model %s", file)
        elif file.suffix == ".keras":
            try:
                models[file.stem] = keras.models.load_model(file)
            except Exception:
                logger.error("Failed to load model %s", file)
    if not models:
        raise FileNotFoundError(
            f"No trained models found in {model_dir}. Run the monthly training first."
        )
    return models


def run_predictions(models: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
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
        logger.info("Using target column %s for %s", target_col, ticker)
        log_df_details(f"predict data {ticker}", df)
        X = df.drop(columns=[target_col], errors="ignore")
        y = df.get(target_col)
        try:
            preds = getattr(model, "predict", lambda X: None)(X)
            if preds is None:
                continue
            mae = mean_absolute_error(y, preds)
            r2 = r2_score(y, preds)
            pred_array = np.asarray(preds).reshape(-1)
            last_pred = pred_array[-1] if pred_array.size else None
            rows.append({
                "ticker": ticker,
                "mae": mae,
                "r2": r2,
                "actual": y.iloc[-1],
                "pred": last_pred,
            })
        except Exception:
            logger.error("Prediction failed for %s", name)
    result_df = pd.DataFrame(rows)
    out_file = RESULTS_DIR / "predictions.csv"
    try:
        result_df.to_csv(out_file, index=False)
        logger.info("Saved predictions to %s", out_file)
    except Exception:
        logger.error("Failed to save predictions to %s", out_file)
    log_offline_mode("prediction")
    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    from .abt.build_abt import build_abt

    data_paths = build_abt()
    data = {t: pd.read_csv(p, index_col=0, parse_dates=True) for t, p in data_paths.items()}
    models = load_models(MODEL_DIR)
    run_predictions(models, data)
