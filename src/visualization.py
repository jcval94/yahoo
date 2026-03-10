import json
import logging
import shutil
from base64 import b64decode
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

try:  # Optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - pandas may not be installed
    pd = None  # type: ignore

try:  # Optional dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib may not be installed
    plt = None  # type: ignore


def load_config(path: Path) -> dict:
    """Load config.yaml with yaml or a simple fallback parser."""
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            return loaded if isinstance(loaded, dict) else {}
    except Exception:
        pass

    config: dict = {}
    current_key = None
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":") and not line.startswith("-"):
                current_key = line[:-1]
                config[current_key] = []
                continue
            if line.startswith("- ") and isinstance(config.get(current_key), list):
                config[current_key].append(line[2:].strip().strip('"').strip("'"))
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                config[key] = value
    return config


def log_df_details(name: str, df, head: int = 3) -> None:
    """Minimal logger replacement."""
    logger.info("%s generated", name)


logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
CONFIG = load_config(CONFIG_PATH)

DATA_DIR = Path(__file__).resolve().parents[1] / CONFIG.get("data_dir", "data")
PRED_DIR = Path(__file__).resolve().parents[1] / "results" / "predicts"
FEATURE_DIR = Path(__file__).resolve().parents[1] / "results" / "features"
METRICS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
EDGE_METRICS_DIR = Path(__file__).resolve().parents[1] / "results" / "edge_metrics"
ACTIONS_DIR = Path(__file__).resolve().parents[1] / "results" / "actions"
VIZ_DIR = Path(__file__).resolve().parents[1] / "results" / "viz"
VIZ_DIR.mkdir(exist_ok=True, parents=True)
DOCS_VIZ_DIR = Path(__file__).resolve().parents[1] / "docs" / "viz"
DOCS_VIZ_DIR.mkdir(exist_ok=True, parents=True)

VIZ_WINDOW_DAYS = 120

PLACEHOLDER_PNG = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/5+BFwAI/AL+WktqAAAAAElFTkSuQmCC"
)


def _write_placeholder(out_file: Path, label: str) -> None:
    """Create simple placeholder PNG and SVG files."""
    out_file.write_bytes(PLACEHOLDER_PNG)
    svg_path = out_file.with_suffix(".svg")
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='120'>"
        "<rect width='300' height='120' fill='#eeeeee'/><text x='150' y='60' "
        "dominant-baseline='middle' text-anchor='middle' font-size='20'>"
        f"{label}</text></svg>"
    )
    svg_path.write_text(svg, encoding="utf-8")


def _latest_csv(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def _latest_csv_for_run_date(directory: Path, prefix: str, run_date: str) -> Path | None:
    """Return latest dated CSV for a step, preferring run_date or the nearest earlier date."""
    if pd is None:
        return _latest_csv(directory, f"{prefix}*.csv")

    target_date = pd.to_datetime(run_date, errors="coerce")
    if pd.isna(target_date):
        return _latest_csv(directory, f"{prefix}*.csv")

    exact = directory / f"{prefix}{run_date}.csv"
    if exact.exists():
        return exact

    candidates: list[tuple[object, Path]] = []
    for candidate in directory.glob(f"{prefix}*.csv"):
        date_part = candidate.stem.removeprefix(prefix)
        parsed = pd.to_datetime(date_part, errors="coerce")
        if pd.isna(parsed):
            continue
        candidates.append((parsed, candidate))

    if not candidates:
        return None

    previous = [row for row in candidates if row[0] <= target_date]
    if previous:
        previous.sort(key=lambda row: row[0])
        return previous[-1][1]

    candidates.sort(key=lambda row: row[0])
    return candidates[0][1]


def _safe_read_csv(path: Path, **kwargs):
    """Read CSV returning empty DataFrame when file has no parsable columns."""
    if pd is None:
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        logger.warning("Skipping empty CSV file: %s", path)
        return pd.DataFrame()


def _parse_pred_value(value: object) -> float | None:
    """Convert prediction values to float handling legacy formats."""
    if pd is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    cleaned = str(value).strip().strip("[]")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _window_start(max_date, days: int = VIZ_WINDOW_DAYS):
    """Compute start date for a fixed analytics window."""
    if pd is None:
        return None
    return pd.to_datetime(max_date) - pd.Timedelta(days=days)


def prepare_candlestick_data(n_days: int = 15) -> "pd.DataFrame | None":
    """Return OHLC data for the last ``n_days`` of all tickers."""
    if pd is None:
        return None
    frames: List[pd.DataFrame] = []
    for ticker in CONFIG.get("etfs", []):
        path = DATA_DIR / f"{ticker}.csv"
        if not path.exists():
            logger.info("Data file %s not found", path)
            continue
        df = _safe_read_csv(path, parse_dates=["Date"])
        if df is None or df.empty:
            continue
        subset = df[["Date", "Open", "High", "Low", "Close"]].tail(n_days).copy()
        subset["ticker"] = ticker
        frames.append(subset)

    if not frames:
        # Fallback: use latest daily prediction files to keep the chart dynamic.
        fallback_rows: List[dict] = []
        for pred_file in sorted(PRED_DIR.glob("*_daily_predictions.csv"), reverse=True)[: n_days * 3]:
            preds = _safe_read_csv(pred_file)
            if preds is None or preds.empty:
                continue
            required = {"ticker", "actual", "Predicted"}
            if preds.empty or not required.issubset(preds.columns):
                continue
            rows = preds[["ticker", "actual", "Predicted"]].dropna().copy()
            if rows.empty:
                continue
            rows["Date"] = pd.to_datetime(rows["Predicted"], errors="coerce")
            rows = rows.dropna(subset=["Date"])
            rows["Close"] = pd.to_numeric(rows["actual"], errors="coerce")
            rows = rows.dropna(subset=["Close"])
            for _, row in rows.iterrows():
                fallback_rows.append(
                    {
                        "Date": row["Date"],
                        "Open": row["Close"],
                        "High": row["Close"],
                        "Low": row["Close"],
                        "Close": row["Close"],
                        "ticker": str(row["ticker"]),
                    }
                )

        if fallback_rows:
            fallback_df = pd.DataFrame(fallback_rows)
            fallback_df = (
                fallback_df.sort_values("Date")
                .groupby("ticker", as_index=False)
                .tail(n_days)
            )
            frames = [fallback_df]

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_file = VIZ_DIR / "candlestick.csv"
    result.to_csv(out_file, index=False)
    logger.info("Saved candlestick data to %s", out_file)
    log_df_details("candlestick data", result)
    return result


def prepare_pred_vs_real(max_files: int = 120) -> "pd.DataFrame | None":
    """Return table comparing predictions with actual prices."""
    if pd is None:
        return None

    rows: List[Dict[str, float]] = []
    daily_files = sorted(PRED_DIR.glob("*_daily_predictions.csv"), reverse=True)

    for pred_file in daily_files[:max_files]:
        preds = _safe_read_csv(pred_file)
        if preds is None or preds.empty:
            continue
        required = {"ticker", "model", "pred", "actual", "Predicted"}
        if preds.empty or not required.issubset(preds.columns):
            continue
        for _, r in preds.iterrows():
            pred_val = _parse_pred_value(r.get("pred"))
            real_val = _parse_pred_value(r.get("actual"))
            if pred_val is None or real_val is None:
                continue
            rows.append(
                {
                    "ticker": str(r["ticker"]),
                    "model": str(r["model"]),
                    "pred": float(pred_val),
                    "real": float(real_val),
                    "Predicted": str(r["Predicted"]),
                }
            )

    result = pd.DataFrame(rows, columns=["ticker", "model", "pred", "real", "Predicted"])
    if not result.empty:
        result["Predicted"] = pd.to_datetime(result["Predicted"], errors="coerce")
        result = result.dropna(subset=["Predicted"])
        start = _window_start(result["Predicted"].max())
        result = result[result["Predicted"] >= start]
        result["Predicted"] = result["Predicted"].dt.strftime("%Y-%m-%d")

    out_file = VIZ_DIR / "pred_vs_real.csv"
    result.to_csv(out_file, index=False)
    logger.info("Saved prediction comparison to %s", out_file)
    log_df_details("pred vs real", result)
    return result


def prepare_best_variables(top_n: int = 10) -> "pd.DataFrame | None":
    """Return the top features ranked by average importance in a fixed window."""
    if pd is None:
        return None
    files = sorted(FEATURE_DIR.glob("features_*_*.csv"), reverse=True)
    if not files:
        logger.info("No feature importance file found")
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for file in files[:120]:
        df = _safe_read_csv(file)
        if df is None or df.empty:
            continue
        if df.empty or not {"feature", "importance_mean", "run_date"}.issubset(df.columns):
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    feat_df = pd.concat(frames, ignore_index=True)
    feat_df["run_date"] = pd.to_datetime(feat_df["run_date"], errors="coerce")
    feat_df = feat_df.dropna(subset=["run_date"])
    start = _window_start(feat_df["run_date"].max())
    feat_df = feat_df[feat_df["run_date"] >= start]

    agg = (
        feat_df.groupby("feature")["importance_mean"]
        .mean()
        .reset_index(name="importance_mean")
    )
    top_df = agg.sort_values("importance_mean", ascending=False).head(top_n)
    out_file = VIZ_DIR / "best_variables.csv"
    top_df.to_csv(out_file, index=False)
    logger.info("Saved best variables to %s", out_file)
    log_df_details("best variables", top_df)
    return top_df


def prepare_edge_metrics(max_files: int = 120) -> "pd.DataFrame | None":
    """Load and normalize recent edge metrics for analytics plots."""
    if pd is None:
        return None

    files = sorted(EDGE_METRICS_DIR.glob("edge_metrics_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for file in files[:max_files]:
        df = _safe_read_csv(file)
        if df is None or df.empty:
            continue
        required = {
            "ticker",
            "model",
            "pred",
            "real",
            "pred_inc",
            "Predicted",
            "MAE",
            "MAPE",
            "RMSE",
            "direction",
        }
        if df.empty or not required.issubset(df.columns):
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    metrics = pd.concat(frames, ignore_index=True)
    metrics["Predicted"] = pd.to_datetime(metrics["Predicted"], errors="coerce")
    metrics = metrics.dropna(subset=["Predicted"])
    start = _window_start(metrics["Predicted"].max())
    metrics = metrics[metrics["Predicted"] >= start]

    for col in ["pred", "real", "pred_inc", "MAE", "MAPE", "RMSE"]:
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    metrics["residual"] = metrics["pred"] - metrics["real"]
    metrics["direction"] = metrics["direction"].astype(str).str.lower().isin(["true", "1"])
    metrics = metrics.dropna(subset=["pred", "real", "MAE", "MAPE", "RMSE", "pred_inc"])

    out_file = VIZ_DIR / "edge_metrics_analytics.csv"
    metrics.to_csv(out_file, index=False)
    logger.info("Saved normalized edge metrics to %s", out_file)
    log_df_details("edge metrics analytics", metrics)
    return metrics


def prepare_feature_stability(max_files: int = 120) -> "pd.DataFrame | None":
    """Prepare feature importance over time for driver stability view."""
    if pd is None:
        return None

    files = sorted(FEATURE_DIR.glob("features_*_*.csv"), reverse=True)
    frames: List[pd.DataFrame] = []
    for file in files[:max_files]:
        df = _safe_read_csv(file)
        if df is None or df.empty:
            continue
        required = {"model", "feature", "importance_mean", "run_date"}
        if df.empty or not required.issubset(df.columns):
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    feat_df = pd.concat(frames, ignore_index=True)
    feat_df["run_date"] = pd.to_datetime(feat_df["run_date"], errors="coerce")
    feat_df["importance_mean"] = pd.to_numeric(feat_df["importance_mean"], errors="coerce")
    feat_df = feat_df.dropna(subset=["run_date", "importance_mean"])
    start = _window_start(feat_df["run_date"].max())
    feat_df = feat_df[feat_df["run_date"] >= start]
    feat_df["run_date"] = feat_df["run_date"].dt.strftime("%Y-%m-%d")

    out_file = VIZ_DIR / "feature_stability.csv"
    feat_df.to_csv(out_file, index=False)
    logger.info("Saved feature stability base data to %s", out_file)
    return feat_df


def prepare_operational_coverage(max_files: int = 120) -> "pd.DataFrame | None":
    """Prepare operational coverage by model/date from daily predictions."""
    if pd is None:
        return None

    files = sorted(PRED_DIR.glob("*_daily_predictions.csv"), reverse=True)
    frames: List[pd.DataFrame] = []
    for file in files[:max_files]:
        df = _safe_read_csv(file)
        if df is None or df.empty:
            continue
        required = {"ticker", "model", "Predicted", "pred"}
        if df.empty or not required.issubset(df.columns):
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    preds = pd.concat(frames, ignore_index=True)
    preds["Predicted"] = pd.to_datetime(preds["Predicted"], errors="coerce")
    preds = preds.dropna(subset=["Predicted"])
    start = _window_start(preds["Predicted"].max())
    preds = preds[preds["Predicted"] >= start]

    total_tickers = max(len(CONFIG.get("etfs", [])), preds["ticker"].nunique())
    valid = preds[preds["pred"].notna()].copy()
    coverage = (
        valid.groupby(["Predicted", "model"], as_index=False)["ticker"]
        .nunique()
        .rename(columns={"ticker": "covered_tickers"})
    )
    coverage["coverage_pct"] = (coverage["covered_tickers"] / total_tickers) * 100.0
    coverage["Predicted"] = coverage["Predicted"].dt.strftime("%Y-%m-%d")

    out_file = VIZ_DIR / "operational_coverage.csv"
    coverage.to_csv(out_file, index=False)
    logger.info("Saved operational coverage data to %s", out_file)
    return coverage


def prepare_strategy_performance() -> "pd.DataFrame | None":
    """Prepare latest backtest performance for the five trading strategies."""
    if pd is None:
        return None

    files = sorted(ACTIONS_DIR.glob("strategy_backtest_*d_summary.csv"), reverse=True)
    required_cols = [
        "strategy",
        "ending_equity",
        "return_pct",
        "win_rate",
        "max_drawdown",
        "initial_budget",
        "max_position_pct",
        "min_trade_usd",
        "holding_days",
    ]
    strategy_names = [
        "winner_take_all",
        "top3_ensemble",
        "consensus_vote",
        "risk_adjusted_edge",
        "downtrend_rebound",
    ]
    out_file = VIZ_DIR / "strategy_performance.csv"

    def _default_output() -> "pd.DataFrame":
        return pd.DataFrame(
            [
                {
                    "strategy": name,
                    "ending_equity": 0.0,
                    "return_pct": 0.0,
                    "win_rate": 0.0,
                    "max_drawdown": 0.0,
                    "initial_budget": float(CONFIG.get("actions", {}).get("initial_budget", 0.0) or 0.0),
                    "max_position_pct": float(CONFIG.get("actions", {}).get("max_position_pct", 0.0) or 0.0) * 100.0,
                    "min_trade_usd": float(CONFIG.get("actions", {}).get("min_trade_usd", 0.0) or 0.0),
                    "holding_days": int(CONFIG.get("actions", {}).get("holding_days", 0) or 0),
                }
                for name in strategy_names
            ],
            columns=required_cols,
        )

    if not files:
        out = _default_output()
        out.to_csv(out_file, index=False)
        return out

    latest = files[0]
    summary = _safe_read_csv(latest)
    minimal_required = {"strategy"}
    if summary is None or summary.empty or not minimal_required.issubset(summary.columns):
        out = _default_output()
        out.to_csv(out_file, index=False)
        return out

    summary = summary.copy()
    if "ending_equity" not in summary.columns and "final_equity" in summary.columns:
        summary["ending_equity"] = summary["final_equity"]
    if "return_pct" not in summary.columns and "total_return_pct" in summary.columns:
        summary["return_pct"] = summary["total_return_pct"]
    if "win_rate" not in summary.columns and "win_rate_pct" in summary.columns:
        summary["win_rate"] = summary["win_rate_pct"]
    if "max_drawdown" not in summary.columns and "max_drawdown_pct" in summary.columns:
        summary["max_drawdown"] = summary["max_drawdown_pct"] / 100.0
    if "initial_budget" not in summary.columns:
        summary["initial_budget"] = float(CONFIG.get("actions", {}).get("initial_budget", 0.0) or 0.0)
    if "max_position_pct" not in summary.columns:
        summary["max_position_pct"] = float(CONFIG.get("actions", {}).get("max_position_pct", 0.0) or 0.0) * 100.0
    if "min_trade_usd" not in summary.columns:
        summary["min_trade_usd"] = float(CONFIG.get("actions", {}).get("min_trade_usd", 0.0) or 0.0)
    if "holding_days" not in summary.columns:
        summary["holding_days"] = int(CONFIG.get("actions", {}).get("holding_days", 0) or 0)

    out = summary[required_cols].copy()
    out["ending_equity"] = pd.to_numeric(out["ending_equity"], errors="coerce").fillna(0.0).round(2)
    out["return_pct"] = pd.to_numeric(out["return_pct"], errors="coerce").fillna(0.0).round(2)
    out["win_rate"] = pd.to_numeric(out["win_rate"], errors="coerce").fillna(0.0).round(2)
    out["max_drawdown"] = pd.to_numeric(out["max_drawdown"], errors="coerce").fillna(0.0).round(4)
    out["initial_budget"] = pd.to_numeric(out["initial_budget"], errors="coerce").fillna(0.0).round(2)
    out["max_position_pct"] = pd.to_numeric(out["max_position_pct"], errors="coerce").fillna(0.0).round(2)
    out["min_trade_usd"] = pd.to_numeric(out["min_trade_usd"], errors="coerce").fillna(0.0).round(2)
    out["holding_days"] = pd.to_numeric(out["holding_days"], errors="coerce").fillna(0).astype(int)

    existing_strategies = set(out["strategy"].dropna().astype(str))
    missing_strategies = [name for name in strategy_names if name not in existing_strategies]
    if missing_strategies:
        filler = pd.DataFrame(
            [
                {
                    "strategy": name,
                    "ending_equity": 0.0,
                    "return_pct": 0.0,
                    "win_rate": 0.0,
                    "max_drawdown": 0.0,
                    "initial_budget": float(CONFIG.get("actions", {}).get("initial_budget", 0.0) or 0.0),
                    "max_position_pct": float(CONFIG.get("actions", {}).get("max_position_pct", 0.0) or 0.0) * 100.0,
                    "min_trade_usd": float(CONFIG.get("actions", {}).get("min_trade_usd", 0.0) or 0.0),
                    "holding_days": int(CONFIG.get("actions", {}).get("holding_days", 0) or 0),
                }
                for name in missing_strategies
            ]
        )
        out = pd.concat([out, filler], ignore_index=True)

    out = out.sort_values("return_pct", ascending=False).head(5)

    out.to_csv(out_file, index=False)
    logger.info("Saved strategy performance data to %s", out_file)
    return out


def prepare_pipeline_health() -> "pd.DataFrame | None":
    """Prepare a compact health summary for the latest pipeline run."""
    if pd is None:
        return None

    evaluation_dir = EDGE_METRICS_DIR
    required_steps = 5
    latest_pred_file = _latest_csv(PRED_DIR, "*_daily_predictions.csv")
    out_file = VIZ_DIR / "pipeline_health.csv"

    if latest_pred_file is None:
        empty = pd.DataFrame(
            [
                {
                    "run_date": "n/a",
                    "duration_minutes": "n/a",
                    "success_pct": 0.0,
                    "successful_steps": 0,
                    "total_steps": required_steps,
                    "fallback_offline": "Sin datos",
                    "status": "SIN EJECUCIONES",
                }
            ]
        )
        empty.to_csv(out_file, index=False)
        return empty

    run_date = latest_pred_file.stem.split("_")[0]
    step_paths: dict[str, Path | None] = {
        "features": _latest_csv_for_run_date(FEATURE_DIR, "features_daily_", run_date),
        "training": _latest_csv_for_run_date(METRICS_DIR, "metrics_daily_", run_date),
        "prediction": _latest_csv(PRED_DIR, f"{run_date}_daily_predictions.csv"),
        "evaluation": _latest_csv_for_run_date(evaluation_dir, "edge_metrics_", run_date),
        "dashboard": VIZ_DIR / "manifest.json" if (VIZ_DIR / "manifest.json").exists() else None,
    }

    successful_steps = sum(1 for p in step_paths.values() if p is not None and p.exists())
    success_pct = round((successful_steps / required_steps) * 100.0, 2)

    mtimes = [p.stat().st_mtime for p in step_paths.values() if p is not None and p.exists()]
    if len(mtimes) >= 2:
        duration_minutes: float | str = round((max(mtimes) - min(mtimes)) / 60.0, 2)
    else:
        duration_minutes = "n/a"

    pred_df = _safe_read_csv(latest_pred_file)
    fallback_offline = "No detectado"
    if pred_df is None or pred_df.empty:
        fallback_offline = "Posible fallback (predicción vacía)"
    elif "actual" in pred_df.columns:
        actual = pd.to_numeric(pred_df["actual"], errors="coerce").dropna()
        if not actual.empty and actual.max() <= 200 and actual.min() >= 0:
            fallback_offline = "Posible modo offline"

    if success_pct >= 90:
        status = "SALUDABLE"
    elif success_pct >= 60:
        status = "DEGRADADO"
    else:
        status = "CRÍTICO"

    out = pd.DataFrame(
        [
            {
                "run_date": run_date,
                "duration_minutes": duration_minutes,
                "success_pct": success_pct,
                "successful_steps": successful_steps,
                "total_steps": required_steps,
                "fallback_offline": fallback_offline,
                "status": status,
            }
        ]
    )
    out.to_csv(out_file, index=False)
    logger.info("Saved pipeline health data to %s", out_file)
    return out


def prepare_action_recommendations() -> "pd.DataFrame | None":
    """Prepare historical per-ticker recommendations and post-action outcomes."""
    if pd is None:
        return None

    try:
        from .actions.paper_trader import (
            _evaluate_strategies,
            _load_latest_model_scores,
            _load_prediction_files,
            _load_stability_scores,
            load_trading_config,
        )
    except Exception:
        return pd.DataFrame()

    preds = _load_prediction_files()
    if preds.empty:
        return pd.DataFrame()

    cfg = load_trading_config()
    model_scores, ranked_models = _load_latest_model_scores()
    stability_scores = _load_stability_scores()

    preds = preds.sort_values(["predicted_date", "ticker", "model"]).copy()
    model_names = sorted(preds["model"].dropna().astype(str).unique())

    closes = (
        preds[["ticker", "predicted_date", "actual"]]
        .dropna(subset=["ticker", "predicted_date", "actual"])
        .drop_duplicates(subset=["ticker", "predicted_date"], keep="last")
        .sort_values(["ticker", "predicted_date"])
    )

    close_map: dict[tuple[str, str], float] = {}
    ticker_dates: dict[str, list[pd.Timestamp]] = {}
    for ticker, grp in closes.groupby("ticker"):
        dates = list(pd.to_datetime(grp["predicted_date"]).dt.normalize())
        ticker_dates[str(ticker)] = dates
        for _, row in grp.iterrows():
            close_map[(str(row["ticker"]), pd.Timestamp(row["predicted_date"]).date().isoformat())] = float(row["actual"])

    def _future_return(ticker: str, date_value: pd.Timestamp, horizon: int) -> float | None:
        dates = ticker_dates.get(ticker, [])
        if not dates:
            return None
        date_value = pd.Timestamp(date_value).normalize()
        if date_value not in dates:
            return None
        pos = dates.index(date_value)
        future_pos = pos + horizon
        if future_pos >= len(dates):
            return None
        start_key = (ticker, date_value.date().isoformat())
        future_key = (ticker, dates[future_pos].date().isoformat())
        start_px = close_map.get(start_key)
        future_px = close_map.get(future_key)
        if start_px is None or future_px is None or start_px == 0:
            return None
        return (future_px / start_px) - 1

    def _format_pct(value: float | None) -> str:
        if value is None:
            return "N/D"
        return f"{value * 100:.2f}%"

    def _outcome_label(action: str, ret: float | None) -> str:
        if ret is None:
            return "Pendiente"
        if action == "BUY":
            return "Acierto" if ret > 0 else "Fallo"
        if action == "SELL":
            return "Acierto" if ret < 0 else "Fallo"
        return "Acierto" if abs(ret) <= 0.01 else "Fallo"

    def _direction(pred: float, actual: float) -> str:
        delta = pred - actual
        if delta > 1e-6:
            return "SUBE"
        if delta < -1e-6:
            return "BAJA"
        return "NEUTRO"

    date_cutoff = preds["predicted_date"].max() - pd.Timedelta(days=60)
    relevant_preds = preds[preds["predicted_date"] >= date_cutoff]

    rows: list[dict] = []
    for (date_value, _ticker), grp in relevant_preds.groupby(["predicted_date", "ticker"]):
        decision = _evaluate_strategies(grp, model_scores, ranked_models, stability_scores)
        score = float(decision["score"])
        if score >= cfg.buy_threshold:
            action = "BUY"
        elif score <= cfg.sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        grp_by_model = grp.groupby("model", as_index=True)["pred"].mean()
        model_predictions = {
            f"pred_{model}": (
                _direction(float(grp_by_model.loc[model]), float(decision["actual"]))
                if model in grp_by_model.index
                else "-"
            )
            for model in model_names
        }

        ret_1d = _future_return(decision["ticker"], pd.Timestamp(date_value), 1)
        ret_5d = _future_return(decision["ticker"], pd.Timestamp(date_value), 5)
        ret_20d = _future_return(decision["ticker"], pd.Timestamp(date_value), 20)

        rows.append(
            {
                "date": pd.Timestamp(date_value).date().isoformat(),
                "ticker": decision["ticker"],
                "best_model": decision["best_model"],
                "strategy_score": round(score, 4),
                "action": action,
                "ret_1d": _format_pct(ret_1d),
                "ret_5d": _format_pct(ret_5d),
                "ret_20d": _format_pct(ret_20d),
                "result_5d": _outcome_label(action, ret_5d),
                **model_predictions,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["date", "strategy_score"], ascending=[False, False])
    base_columns = [
        "date",
        "ticker",
        "best_model",
        "strategy_score",
        "action",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "result_5d",
    ]
    model_columns = [f"pred_{model}" for model in model_names]
    out = out.reindex(columns=base_columns + model_columns)
    out_file = VIZ_DIR / "action_recommendations.csv"
    out.to_csv(out_file, index=False)
    logger.info("Saved action recommendations data to %s", out_file)
    return out



def prepare_last_run_report(
    health_df: "pd.DataFrame | None",
    action_df: "pd.DataFrame | None",
    strategy_df: "pd.DataFrame | None",
) -> dict | None:
    """Build a comprehensive report payload for the latest run."""
    if pd is None:
        return None

    report_file = VIZ_DIR / "last_run_report.json"
    latest_pred_file = _latest_csv(PRED_DIR, "*_daily_predictions.csv")
    latest_metrics_file = _latest_csv(METRICS_DIR, "metrics_daily_*.csv")
    latest_edge_file = _latest_csv(EDGE_METRICS_DIR, "edge_metrics_*.csv")

    run_date = latest_pred_file.stem.split("_")[0] if latest_pred_file else "n/a"

    health_row = {}
    if health_df is not None and not getattr(health_df, "empty", True):
        health_row = health_df.iloc[0].to_dict()

    action_slice = pd.DataFrame()
    action_selection = "none"
    action_dates = pd.Series(dtype="datetime64[ns]")
    selected_action_date = "n/a"
    if action_df is not None and not getattr(action_df, "empty", True):
        if "date" in action_df.columns:
            action_dates = pd.to_datetime(action_df["date"], errors="coerce")
            if run_date != "n/a":
                action_slice = action_df[action_df["date"] == run_date].copy()
                if not action_slice.empty:
                    action_selection = "run_date_exact"
                    selected_action_date = run_date
            if action_slice.empty:
                max_date = action_dates.max()
                if not pd.isna(max_date):
                    action_slice = action_df[action_dates == max_date].copy()
                    if not action_slice.empty:
                        action_selection = "max_date_fallback"
                        selected_action_date = pd.Timestamp(max_date).date().isoformat()
            if run_date == "n/a" and action_slice.empty:
                action_slice = action_df.copy()
                action_selection = "run_date_missing_all_actions"
        else:
            action_slice = action_df.copy()
            action_selection = "date_column_missing"

    strategy_rows: list[dict] = []
    if strategy_df is not None and not getattr(strategy_df, "empty", True):
        strategy_rows = strategy_df.head(5).to_dict(orient="records")

    action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    quality_counts = {"Acierto": 0, "Fallo": 0, "Pendiente": 0}
    top_recommendations: list[dict] = []
    if not action_slice.empty:
        action_counts.update(action_slice["action"].value_counts().to_dict())
        if "result_5d" in action_slice.columns:
            quality_counts.update(action_slice["result_5d"].value_counts().to_dict())
        top_recommendations = (
            action_slice.sort_values("strategy_score", ascending=False)
            .head(10)
            [["ticker", "action", "strategy_score", "ret_1d", "ret_5d", "ret_20d", "result_5d"]]
            .to_dict(orient="records")
        )

    metrics_summary: list[dict] = []
    if latest_metrics_file is not None:
        mdf = _safe_read_csv(latest_metrics_file)
        if mdf is not None and not mdf.empty and {"model", "MAE"}.issubset(mdf.columns):
            work = mdf.copy()
            if "dataset" in work.columns:
                work = work[work["dataset"].astype(str).str.lower() == "test"]
            work["MAE"] = pd.to_numeric(work["MAE"], errors="coerce")
            work = work.dropna(subset=["MAE"]).sort_values("MAE")
            metrics_summary = work.head(10)[[c for c in ["model", "MAE", "RMSE", "MAPE", "R2"] if c in work.columns]].to_dict(orient="records")

    edge_summary = {"rows": 0, "tickers": 0, "models": 0, "by_model": []}
    if latest_edge_file is not None:
        edf = _safe_read_csv(latest_edge_file)
        if edf is not None and not edf.empty:
            by_model: list[dict] = []
            if {"model", "ticker"}.issubset(edf.columns):
                grouped = (
                    edf.groupby("model", as_index=False)
                    .agg(rows=("ticker", "size"), tickers=("ticker", "nunique"))
                    .sort_values(["tickers", "rows", "model"], ascending=[False, False, True])
                )
                by_model = [
                    {
                        "model": str(row["model"]),
                        "rows": int(row["rows"]),
                        "tickers": int(row["tickers"]),
                    }
                    for _, row in grouped.iterrows()
                ]

            edge_summary = {
                "rows": int(len(edf)),
                "tickers": int(edf["ticker"].nunique()) if "ticker" in edf.columns else 0,
                "models": int(edf["model"].nunique()) if "model" in edf.columns else 0,
                "by_model": by_model,
            }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "artifacts": {
            "predictions_file": latest_pred_file.name if latest_pred_file else "n/a",
            "metrics_file": latest_metrics_file.name if latest_metrics_file else "n/a",
            "edge_metrics_file": latest_edge_file.name if latest_edge_file else "n/a",
        },
        "pipeline_health": health_row,
        "summary": {
            "actions": action_counts,
            "quality_5d": quality_counts,
            "recommended_tickers": int(len(action_slice)),
            "strategies_ranked": int(len(strategy_rows)),
            "edge_coverage": edge_summary,
        },
        "top_recommendations": top_recommendations,
        "strategy_leaderboard": strategy_rows,
        "model_metrics": metrics_summary,
    }

    action_dates_clean = action_dates.dropna() if action_dates is not None else pd.Series(dtype="datetime64[ns]")
    logger.info(
        "last_run_report_snapshot run_date=%s action_selection=%s action_rows_total=%s "
        "action_rows_selected=%s action_date_min=%s action_date_max=%s selected_action_date=%s "
        "top_recommendations=%s strategy_rows=%s metrics_rows=%s edge_rows=%s edge_tickers=%s edge_models=%s edge_by_model=%s",
        run_date,
        action_selection,
        0 if action_df is None else int(len(action_df)),
        int(len(action_slice)),
        "n/a" if action_dates_clean.empty else action_dates_clean.min().date().isoformat(),
        "n/a" if action_dates_clean.empty else action_dates_clean.max().date().isoformat(),
        selected_action_date,
        len(top_recommendations),
        len(strategy_rows),
        len(metrics_summary),
        edge_summary["rows"],
        edge_summary["tickers"],
        edge_summary["models"],
        len(edge_summary.get("by_model", [])),
    )

    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved last run report to %s", report_file)
    return report

def _plot_candlestick(df: "pd.DataFrame | None", out_file: Path) -> None:
    """Create line plot of closing prices."""
    if pd is None or plt is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "Candlestick")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    norm = df.copy()
    norm["Date"] = pd.to_datetime(norm["Date"], errors="coerce")
    norm = norm.dropna(subset=["Date", "ticker", "Close"])
    pivot = norm.groupby(["Date", "ticker"], as_index=False)["Close"].mean().pivot(index="Date", columns="ticker", values="Close")
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(ax=ax, linewidth=1.5)
    ax.set_title("Precio de cierre - últimos 15 días", fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Cierre")
    ax.grid(True, linestyle="--", alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=max(1, len(labels)), frameon=False)
    fig.tight_layout(pad=2)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".svg"))
    plt.close(fig)


def _plot_pred_vs_real(df: "pd.DataFrame | None", out_file: Path) -> None:
    """Scatter plot of predicted vs real prices."""
    if pd is None or plt is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "Pred vs Real")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["real"], df["pred"], alpha=0.7, edgecolor="k", s=40)
    min_val = min(df["real"].min(), df["pred"].min())
    max_val = max(df["real"].max(), df["pred"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Ideal")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicción")
    ax.set_title("Predicción vs Real", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=2)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".svg"))
    plt.close(fig)


def _plot_best_variables(df: "pd.DataFrame | None", out_file: Path) -> None:
    """Bar plot of feature importance."""
    if pd is None or plt is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "Best Variables")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="barh", x="feature", y="importance_mean", ax=ax, color="#348ABD")
    ax.invert_yaxis()
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    ax.set_title("Variables más importantes", fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    legend = ax.get_legend()
    if legend:
        legend.remove()
    fig.tight_layout(pad=2)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".svg"))
    plt.close(fig)


def _plot_error_heatmap(df: "pd.DataFrame | None", out_file: Path) -> None:
    """Heatmap of average MAE by ticker and model."""
    if pd is None or plt is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "MAE Heatmap")
        return

    mae_table = df.groupby(["ticker", "model"], as_index=False)["MAE"].mean()
    ticker_order = mae_table.groupby("ticker")["MAE"].mean().sort_values(ascending=False).head(20).index
    pivot = (
        mae_table[mae_table["ticker"].isin(ticker_order)]
        .pivot(index="ticker", columns="model", values="MAE")
        .sort_index()
    )
    if pivot.empty:
        _write_placeholder(out_file, "MAE Heatmap")
        return

    pivot.to_csv(VIZ_DIR / "error_heatmap.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Heatmap de MAE por ticker y modelo", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MAE")
    fig.tight_layout(pad=2)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".svg"))
    plt.close(fig)


def _plot_direction_accuracy(df: "pd.DataFrame | None", out_file: Path) -> None:
    """Heatmap of direction accuracy by ticker and model."""
    if pd is None or plt is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "Direction Accuracy")
        return

    acc = df.groupby(["ticker", "model"], as_index=False)["direction"].mean()
    ticker_order = acc.groupby("ticker")["direction"].mean().sort_values(ascending=True).head(20).index
    pivot = (
        acc[acc["ticker"].isin(ticker_order)]
        .pivot(index="ticker", columns="model", values="direction")
        .sort_index()
    )
    if pivot.empty:
        _write_placeholder(out_file, "Direction Accuracy")
        return

    preferred_model_order = ["Top3Ensamble", "arima", "lgbm", "linreg", "rf", "xgb"]
    ordered_columns = [m for m in preferred_model_order if m in pivot.columns] + [
        m for m in pivot.columns if m not in preferred_model_order
    ]
    pivot = pivot.reindex(columns=ordered_columns)

    (pivot * 100).to_csv(VIZ_DIR / "direction_accuracy.csv")

    model_coverage = (
        acc.groupby("model")["ticker"]
        .nunique()
        .reindex(ordered_columns)
        .fillna(0)
        .astype(int)
    )
    total_tickers = int(acc["ticker"].nunique())
    coverage_note = " | ".join(
        f"{model}: {count}/{total_tickers}" for model, count in model_coverage.items()
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#d9d9d9")
    im = ax.imshow((pivot * 100).to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Accuracy direccional (%) por ticker y modelo", fontweight="bold")
    ax.text(
        0.0,
        -0.2,
        f"Gris = sin datos | Cobertura ticker-modelo: {coverage_note}",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)")
    fig.tight_layout(pad=2.2)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".svg"))
    plt.close(fig)


def _plot_residual_distribution(df: "pd.DataFrame | None", out_file: Path) -> None:
    """Boxplot of residuals by model."""
    if pd is None or plt is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "Residual Distribution")
        return

    tmp = df[["model", "residual"]].dropna()
    grouped = [grp["residual"].values for _, grp in tmp.groupby("model")]
    labels = [model for model, _ in tmp.groupby("model")]
    if not grouped:
        _write_placeholder(out_file, "Residual Distribution")
        return

    tmp.to_csv(VIZ_DIR / "residual_distribution.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(grouped, labels=labels, patch_artist=True)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Distribución de residuales por modelo (pred - real)", fontweight="bold")
    ax.set_ylabel("Residual")
    ax.set_xlabel("Modelo")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(pad=2)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".svg"))
    plt.close(fig)


def _plot_mae_trend(df: "pd.DataFrame | None", out_file: Path) -> None:
    """Temporal MAE trend by model."""
    if pd is None or plt is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "MAE Trend")
        return

    trend = (
        df.groupby(["Predicted", "model"], as_index=False)["MAE"].mean()
        .sort_values("Predicted")
    )
    if trend.empty:
        _write_placeholder(out_file, "MAE Trend")
        return

    trend.to_csv(VIZ_DIR / "mae_trend.csv", index=False)
    pivot = trend.pivot(index="Predicted", columns="model", values="MAE")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(ax=ax, linewidth=1.7)
    ax.set_title("Tendencia temporal de MAE por modelo", fontweight="bold")
    ax.set_xlabel("Fecha de predicción")
    ax.set_ylabel("MAE")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(6, len(pivot.columns)), frameon=False)
    fig.tight_layout(pad=2)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".svg"))
    plt.close(fig)


def _plot_risk_return_scatter(df: "pd.DataFrame | None", out_file: Path) -> None:
    """Risk-return scatter of signal increment vs error."""
    if pd is None or plt is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "Risk Return")
        return

    rr = (
        df.groupby(["ticker", "model"], as_index=False)
        .agg(pred_inc=("pred_inc", "mean"), MAPE=("MAPE", "mean"))
        .dropna()
    )
    if rr.empty:
        _write_placeholder(out_file, "Risk Return")
        return

    rr.to_csv(VIZ_DIR / "risk_return_signals.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, grp in rr.groupby("model"):
        ax.scatter(grp["pred_inc"] * 100, grp["MAPE"] * 100, s=40, alpha=0.72, label=model)
    ax.set_title("Dispersión riesgo-retorno de señales", fontweight="bold")
    ax.set_xlabel("Retorno esperado (pred_inc, %)")
    ax.set_ylabel("Error de señal (MAPE, %)")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    fig.tight_layout(pad=2)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".svg"))
    plt.close(fig)


def _plot_driver_stability(df: "pd.DataFrame | None", out_file: Path, top_n: int = 12) -> None:
    """Heatmap of feature importance stability over time."""
    if pd is None or plt is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "Driver Stability")
        return

    top_features = (
        df.groupby("feature")["importance_mean"]
        .mean()
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    filtered = df[df["feature"].isin(top_features)]
    pivot = (
        filtered.groupby(["feature", "run_date"], as_index=False)["importance_mean"]
        .mean()
        .pivot(index="feature", columns="run_date", values="importance_mean")
    )
    if pivot.empty:
        _write_placeholder(out_file, "Driver Stability")
        return

    pivot.to_csv(VIZ_DIR / "driver_stability.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Estabilidad de drivers (importancia en el tiempo)", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Importance mean")
    fig.tight_layout(pad=2)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".svg"))
    plt.close(fig)


def _plot_operational_coverage(df: "pd.DataFrame | None", out_file: Path) -> None:
    """Coverage curve by model over time."""
    if pd is None or plt is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "Coverage")
        return

    plot_df = df.copy()
    plot_df["Predicted"] = pd.to_datetime(plot_df["Predicted"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Predicted", "coverage_pct"])
    if plot_df.empty:
        _write_placeholder(out_file, "Coverage")
        return

    plot_df.to_csv(VIZ_DIR / "operational_coverage_curve.csv", index=False)
    pivot = plot_df.pivot(index="Predicted", columns="model", values="coverage_pct")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(ax=ax, linewidth=1.8)
    ax.set_title("Curva de cobertura operativa", fontweight="bold")
    ax.set_xlabel("Fecha de predicción")
    ax.set_ylabel("Cobertura de tickers (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=min(6, len(pivot.columns)), frameon=False)
    fig.tight_layout(pad=2)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".svg"))
    plt.close(fig)




def _write_viz_manifest(datasets: List["pd.DataFrame | None"]) -> None:
    """Write metadata so dashboard always references latest generated visuals."""
    if pd is None:
        return

    date_values = []
    for df in datasets:
        if df is None or getattr(df, "empty", True):
            continue
        for col in ["Predicted", "run_date", "Date"]:
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce").dropna()
                if not parsed.empty:
                    date_values.append(parsed.min())
                    date_values.append(parsed.max())

    if date_values:
        window_start = min(date_values).strftime("%Y-%m-%d")
        window_end = max(date_values).strftime("%Y-%m-%d")
    else:
        window_start = "n/a"
        window_end = "n/a"

    now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    manifest = {
        "generated_at": now,
        "window_start": window_start,
        "window_end": window_end,
        "window_days": VIZ_WINDOW_DAYS,
    }

    (VIZ_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved visualization manifest to %s", VIZ_DIR / "manifest.json")

def _copy_viz_files() -> None:
    """Copy generated assets to the docs directory."""
    asset_paths = list(VIZ_DIR.glob("*.svg"))
    asset_paths.extend(
        [
            p
            for p in [
                VIZ_DIR / "strategy_performance.csv",
                VIZ_DIR / "action_recommendations.csv",
                VIZ_DIR / "pipeline_health.csv",
                VIZ_DIR / "last_run_report.json",
            ]
            if p.exists()
        ]
    )
    manifest_file = VIZ_DIR / "manifest.json"
    if manifest_file.exists():
        asset_paths.append(manifest_file)

    for asset_file in asset_paths:
        target = DOCS_VIZ_DIR / asset_file.name
        try:
            shutil.copy2(asset_file, target)
            logger.info("Copied %s to %s", asset_file.name, target)
        except Exception as exc:  # pragma: no cover - safeguard
            logger.warning("Failed to copy %s: %s", asset_file, exc)


def create_viz_tables() -> None:
    """Generate all visualization tables."""
    if plt is None:
        logger.warning("matplotlib is not installed; writing placeholder visualization files")

    candle_df = prepare_candlestick_data()
    pred_df = prepare_pred_vs_real()
    best_df = prepare_best_variables()
    edge_metrics_df = prepare_edge_metrics()
    feature_stability_df = prepare_feature_stability()
    coverage_df = prepare_operational_coverage()
    strategy_df = prepare_strategy_performance()
    health_df = prepare_pipeline_health()
    action_df = prepare_action_recommendations()

    _plot_candlestick(candle_df, VIZ_DIR / "candlestick.png")
    _plot_pred_vs_real(pred_df, VIZ_DIR / "pred_vs_real.png")
    _plot_best_variables(best_df, VIZ_DIR / "best_variables.png")
    _plot_error_heatmap(edge_metrics_df, VIZ_DIR / "error_heatmap.png")
    _plot_direction_accuracy(edge_metrics_df, VIZ_DIR / "direction_accuracy.png")
    _plot_residual_distribution(edge_metrics_df, VIZ_DIR / "residual_distribution.png")
    _plot_mae_trend(edge_metrics_df, VIZ_DIR / "mae_trend.png")
    _plot_risk_return_scatter(edge_metrics_df, VIZ_DIR / "risk_return_signals.png")
    _plot_driver_stability(feature_stability_df, VIZ_DIR / "driver_stability.png")
    _plot_operational_coverage(coverage_df, VIZ_DIR / "operational_coverage.png")
    prepare_last_run_report(health_df, action_df, strategy_df)
    _write_viz_manifest(
        [
            pred_df,
            edge_metrics_df,
            feature_stability_df,
            coverage_df,
            candle_df,
            strategy_df,
            health_df,
            action_df,
        ]
    )
    _copy_viz_files()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_viz_tables()
