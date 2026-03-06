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

REGIME_COLUMNS = ("high_vol_regime", "risk_regime", "session_bucket")


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


def _risk_regime(row: pd.Series) -> str:
    """Infer risk regime from available return/variation fields."""
    candidates = ("real_inc", "pred_inc", "real_delta", "pred_delta")
    value = next((row.get(c) for c in candidates if pd.notna(row.get(c))), None)
    if value is None:
        return "risk_unknown"
    return "risk_on" if float(value) >= 0 else "risk_off"


def enrich_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical market regime columns used for segmented evaluation."""
    if df.empty:
        return df.copy()

    out = df.copy()
    if "session_bucket" not in out.columns:
        if "session" in out.columns:
            out["session_bucket"] = out["session"].fillna("after_hours")
        else:
            ts_col = "timestamp" if "timestamp" in out.columns else "Predicted"
            ts_values = out[ts_col] if ts_col in out.columns else pd.Series([pd.NaT] * len(out), index=out.index)
            out["session_bucket"] = pd.to_datetime(ts_values, errors="coerce").map(_session_from_timestamp)

    if "risk_regime" not in out.columns:
        out["risk_regime"] = out.apply(_risk_regime, axis=1)

    if "high_vol_regime" not in out.columns:
        vol_source = "real_inc" if "real_inc" in out.columns else "pred_inc"
        if vol_source in out.columns and out[vol_source].notna().any():
            abs_move = out[vol_source].astype(float).abs()
            threshold = abs_move.median()
            out["high_vol_regime"] = np.where(abs_move >= threshold, "high_vol_regime", "normal_vol_regime")
            out.loc[abs_move.isna(), "high_vol_regime"] = "vol_unknown"
        else:
            out["high_vol_regime"] = "vol_unknown"

    return out


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

    work = enrich_regime_labels(df)
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


def evaluate_regime_metrics(
    df: pd.DataFrame,
    regime_cols: Sequence[str] = REGIME_COLUMNS,
    baseline_col: str = "baseline_pred",
) -> pd.DataFrame:
    """Compute per-regime metrics and optional comparison against a baseline prediction."""
    required = {"ticker", "model", "pred", "real"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    work = enrich_regime_labels(df)
    records = []
    for regime_col in regime_cols:
        if regime_col not in work.columns:
            continue
        for (ticker, model, regime), grp in work.groupby(["ticker", "model", regime_col], dropna=False):
            metrics = evaluate_predictions(grp["real"], grp["pred"])
            record = {
                "ticker": ticker,
                "model": model,
                "regime_type": regime_col,
                "regime_value": regime,
                "rows": len(grp),
                **metrics,
            }
            if baseline_col in grp.columns and grp[baseline_col].notna().any():
                base_metrics = evaluate_predictions(grp["real"], grp[baseline_col])
                for k, v in base_metrics.items():
                    record[f"baseline_{k}"] = v
                    record[f"delta_vs_baseline_{k}"] = round(metrics[k] - v, 4)
            records.append(record)

    return pd.DataFrame(records)


def save_regime_reports(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Persist per-ticker/model/regime reports for operational monitoring."""
    regime_df = evaluate_regime_metrics(df)
    if regime_df.empty:
        return []

    out_files: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for (ticker, model, regime_type), grp in regime_df.groupby(["ticker", "model", "regime_type"]):
        safe_regime = str(regime_type).replace("/", "_")
        file = output_dir / f"{ticker}_{model}_{safe_regime}.csv"
        grp.to_csv(file, index=False)
        out_files.append(file)
    return out_files
