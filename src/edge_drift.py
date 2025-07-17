"""Evaluate drift based on recent edge metrics."""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import date
from typing import Dict

import pandas as pd

from .evaluation import evaluate_predictions

logger = logging.getLogger(__name__)

# Tolerance thresholds for each metric
TOLERANCE: Dict[str, float] = {
    "MAE": 1.0,
    "MSE": 1.0,
    "RMSE": 1.0,
    "MAPE": 0.05,
    "R2": 0.7,
    "EVS": 0.7,
}


def _load_recent_metrics(directory: Path, days: int = 15) -> pd.DataFrame:
    """Return concatenated edge metrics for the most recent ``days`` files."""
    files = sorted(directory.glob("edge_metrics_*.csv"))
    if not files:
        return pd.DataFrame()
    recent = files[-days:]
    dfs = []
    for file in recent:
        try:
            df = pd.read_csv(file, usecols=["ticker", "model", "pred", "real"])
            dfs.append(df)
        except Exception:
            logger.exception("Failed to load %s", file)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def evaluate_drift(data: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics and drift score for each ticker and model."""
    if data.empty:
        return pd.DataFrame()

    records = []
    for (ticker, model), grp in data.groupby(["ticker", "model"]):
        y_true = grp["real"].tolist()
        y_pred = grp["pred"].tolist()
        metrics = evaluate_predictions(y_true, y_pred)
        drift_hits = 0
        for k, v in metrics.items():
            thr = TOLERANCE.get(k)
            if thr is None:
                continue
            if k in {"R2", "EVS"}:
                if v < thr:
                    drift_hits += 1
            else:
                if v > thr:
                    drift_hits += 1
        drift_score = drift_hits / len(TOLERANCE)
        records.append({"ticker": ticker, "model": model, **metrics, "drift_score": round(drift_score, 4)})

    return pd.DataFrame(records)


def main(days: int = 15) -> Path:
    """Load recent edge metrics and write drift evaluation."""
    base_dir = Path(__file__).resolve().parents[1]
    metrics_dir = base_dir / "results" / "edge_metrics"
    df = _load_recent_metrics(metrics_dir, days)
    result = evaluate_drift(df)
    today = date.today().isoformat()
    out_dir = base_dir / "results" / "drift" / f"edge_drift_evaluation_{today}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "drift_metrics.csv"
    result.to_csv(out_file, index=False)
    logger.info("Saved drift evaluation to %s", out_file)
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
