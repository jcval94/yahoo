"""Model evaluation and drift detection."""
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
    r2_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)

SESSION_BUCKETS = {
    "opening": (9, 30, 11, 30),
    "midday": (11, 30, 15, 0),
    "close": (15, 0, 16, 0),
}


def evaluate_predictions(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    """Return an expanded set of regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "EVS": evs,
    }
    logger.info("Evaluation metrics: %s", metrics)
    return {k: round(v, 4) for k, v in metrics.items()}


def detect_drift(prev: Sequence[float], curr: Sequence[float], threshold: float = 0.1) -> bool:
    """Simple drift detection comparing prediction distributions."""
    prev = np.array(prev)
    curr = np.array(curr)
    diff = np.abs(prev - curr).mean()
    return diff > threshold


def _session_from_timestamp(ts: pd.Timestamp) -> str:
    """Return trading session bucket for a timestamp."""
    if pd.isna(ts):
        return "after_hours"
    ts = pd.Timestamp(ts)
    if ts.tzinfo:
        ts = ts.tz_convert("UTC").tz_localize(None)
    hour_min = (ts.hour, ts.minute)
    for name, (start_h, start_m, end_h, end_m) in SESSION_BUCKETS.items():
        if (start_h, start_m) <= hour_min < (end_h, end_m):
            return name
    return "after_hours"


def _event_type_from_flags(row: pd.Series) -> str:
    """Map calendar/event flags to a canonical event type."""
    earnings_cols = ["is_earnings_day", "earnings_day", "flag_earnings"]
    macro_cols = ["is_macro_day", "macro_day", "flag_macro"]
    if any(bool(row.get(c, False)) for c in earnings_cols):
        return "earnings_day"
    if any(bool(row.get(c, False)) for c in macro_cols):
        return "macro_day"
    return "no_event_day"


def build_segmented_metrics(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Aggregate evaluation metrics by ``segment_col`` and model keys."""
    if df.empty or segment_col not in df.columns:
        return pd.DataFrame()

    records = []
    for (ticker, model, segment), grp in df.groupby(["ticker", "model", segment_col], dropna=False):
        metrics = evaluate_predictions(grp["real"].tolist(), grp["pred"].tolist())
        records.append(
            {
                "ticker": ticker,
                "model": model,
                segment_col: segment,
                "rows": len(grp),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def save_segmented_reports(df: pd.DataFrame, metrics_file: Path) -> tuple[Path | None, Path | None]:
    """Save session and event breakdown metrics alongside ``metrics_file``."""
    if df.empty:
        return None, None

    work = df.copy()
    if "session" not in work.columns:
        ts_col = "timestamp" if "timestamp" in work.columns else "Predicted"
        work["session"] = pd.to_datetime(work.get(ts_col), errors="coerce").map(_session_from_timestamp)
    if "event_type" not in work.columns:
        work["event_type"] = work.apply(_event_type_from_flags, axis=1)

    by_session = build_segmented_metrics(work, "session")
    by_event = build_segmented_metrics(work, "event_type")

    out_dir = Path(metrics_file).parent
    stem = Path(metrics_file).stem
    session_file = out_dir / f"{stem}_by_session.csv"
    event_file = out_dir / f"{stem}_by_event.csv"
    by_session.to_csv(session_file, index=False)
    by_event.to_csv(event_file, index=False)
    logger.info("Saved segmented metrics to %s and %s", session_file, event_file)
    return session_file, event_file
