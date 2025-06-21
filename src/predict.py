"""Apply trained models to new data and store predictions."""
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def load_models(model_dir: Path) -> Dict[str, Any]:
    models = {}
    for file in model_dir.glob("*.pkl"):
        models[file.stem] = joblib.load(file)
    return models


def run_predictions(models: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        ticker = name.split("_")[0]
        df = data.get(ticker)
        if df is None:
            continue
        X = df.drop(columns=["Close"], errors="ignore")
        y = df.get("Close")
        preds = getattr(model, "predict", lambda X: None)(X)
        if preds is None:
            continue
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        rows.append({"ticker": ticker, "mae": mae, "r2": r2, "actual": y.iloc[-1], "pred": preds[-1]})
    result_df = pd.DataFrame(rows)
    out_file = RESULTS_DIR / "predictions.csv"
    result_df.to_csv(out_file, index=False)
    logger.info("Saved predictions to %s", out_file)
    return result_df
