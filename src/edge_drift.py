"""Evaluate drift based on recent edge metrics."""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import date
from typing import Dict

import pandas as pd

from .evaluation import REGIME_COLUMNS, evaluate_predictions, enrich_regime_labels

logger = logging.getLogger(__name__)

# Tolerance thresholds for each metric
BASE_TOLERANCE: Dict[str, float] = {
    "MAE": 1.0,
    "MSE": 1.0,
    "RMSE": 1.0,
    "MAPE": 0.05,
    "R2": 0.7,
    "EVS": 0.7,
}

# Regime-specific tolerance multipliers.
REGIME_MULTIPLIER: Dict[str, Dict[str, float]] = {
    "earnings_day": {"MAE": 1.25, "MSE": 1.4, "RMSE": 1.3, "MAPE": 1.2, "R2": 0.9, "EVS": 0.9},
    "macro_day": {"MAE": 1.15, "MSE": 1.2, "RMSE": 1.15, "MAPE": 1.1, "R2": 0.95, "EVS": 0.95},
    "no_event_day": {"MAE": 1.0, "MSE": 1.0, "RMSE": 1.0, "MAPE": 1.0, "R2": 1.0, "EVS": 1.0},
}


def _regime_for_group(grp: pd.DataFrame) -> str:
    """Infer regime label from available columns in ``grp``."""
    for col in ("event_type", "regime"):
        if col in grp.columns and grp[col].notna().any():
            return str(grp[col].dropna().iloc[0])
    return "no_event_day"


def _tolerance_for_regime(regime: str) -> Dict[str, float]:
    """Return tolerance map adjusted by the supplied regime."""
    mult = REGIME_MULTIPLIER.get(regime, REGIME_MULTIPLIER["no_event_day"])
    out: Dict[str, float] = {}
    for metric, base in BASE_TOLERANCE.items():
        out[metric] = round(base * mult.get(metric, 1.0), 6)
    return out


def _load_recent_metrics(directory: Path, days: int = 15) -> pd.DataFrame:
    """Return concatenated edge metrics for the most recent ``days`` files."""
    files = sorted(directory.glob("edge_metrics_*.csv"))
    if not files:
        return pd.DataFrame()
    recent = files[-days:]
    dfs = []
    for file in recent:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception:
            logger.exception("Failed to load %s", file)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def evaluate_drift(data: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics and drift score for each ticker and model."""
    if data.empty:
        return pd.DataFrame()

    work = enrich_regime_labels(data)
    records = []
    for regime_col in REGIME_COLUMNS:
        if regime_col not in work.columns:
            continue
        for (ticker, model, regime_value), grp in work.groupby(["ticker", "model", regime_col], dropna=False):
            y_true = grp["real"].tolist()
            y_pred = grp["pred"].tolist()
            metrics = evaluate_predictions(y_true, y_pred)
            regime = str(regime_value)
            tolerance = _tolerance_for_regime(regime)
            drift_hits = 0
            for k, v in metrics.items():
                thr = tolerance.get(k)
                if thr is None:
                    continue
                if k in {"R2", "EVS"}:
                    if v < thr:
                        drift_hits += 1
                else:
                    if v > thr:
                        drift_hits += 1
            drift_score = drift_hits / len(tolerance)
            records.append(
                {
                    "ticker": ticker,
                    "model": model,
                    "regime_type": regime_col,
                    "regime": regime,
                    **metrics,
                    "drift_score": round(drift_score, 4),
                }
            )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    summary_keys = ["ticker", "model"]
    failing = result[result["drift_score"] > 0]
    total_by_pair = result.groupby(summary_keys, as_index=False)["regime"].count().rename(columns={"regime": "total_regimes"})
    fail_by_pair = failing.groupby(summary_keys, as_index=False)["regime"].count().rename(columns={"regime": "failed_regimes"})
    result = result.merge(total_by_pair, on=summary_keys, how="left")
    result = result.merge(fail_by_pair, on=summary_keys, how="left")
    result["failed_regimes"] = result["failed_regimes"].fillna(0).astype(int)
    result["regime_only_failure"] = (result["failed_regimes"] > 0) & (result["failed_regimes"] < result["total_regimes"])

    return result


def main(days: int = 15) -> Path:
    """Load recent edge metrics and write drift evaluation."""
    base_dir = Path(__file__).resolve().parents[1]
    metrics_dir = base_dir / "results" / "edge_metrics"
    df = _load_recent_metrics(metrics_dir, days)
    result = evaluate_drift(df)
    today = date.today().isoformat()
    out_file = (
        base_dir
        / "results"
        / "drift"
        / f"edge_drift_evaluation_{today}.csv"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_file, index=False)
    logger.info("Saved drift evaluation to %s", out_file)
    return out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
