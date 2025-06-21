"""Apply trained models to new data and store predictions."""
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score

from .utils import log_df_details, log_offline_mode

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH) as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path(__file__).resolve().parents[1] / CONFIG.get("model_dir", "models")


def load_models(model_dir: Path) -> Dict[str, Any]:
    models = {}
    for file in model_dir.glob("*.pkl"):
        try:
            models[file.stem] = joblib.load(file)
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
        df = data.get(ticker)
        if df is None:
            continue
        log_df_details(f"predict data {ticker}", df)
        X = df.drop(columns=["Close"], errors="ignore")
        y = df.get("Close")
        try:
            preds = getattr(model, "predict", lambda X: None)(X)
            if preds is None:
                continue
            mae = mean_absolute_error(y, preds)
            r2 = r2_score(y, preds)
            rows.append({"ticker": ticker, "mae": mae, "r2": r2, "actual": y.iloc[-1], "pred": preds[-1]})
        except Exception:
            logger.error("Prediction failed for %s", name)
    result_df = pd.DataFrame(rows)
    out_file = RESULTS_DIR / "predictions.csv"
    result_df.to_csv(out_file, index=False)
    logger.info("Saved predictions to %s", out_file)
    log_offline_mode("prediction")
    return result_df


if __name__ == "__main__":
    from .abt.build_abt import build_abt

    data_paths = build_abt()
    data = {t: pd.read_csv(p) for t, p in data_paths.items()}
    models = load_models(MODEL_DIR)
    run_predictions(models, data)
