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
        df = pd.read_csv(path, parse_dates=["Date"])
        subset = df[["Date", "Open", "High", "Low", "Close"]].tail(n_days).copy()
        subset["ticker"] = ticker
        frames.append(subset)

    if not frames:
        # Fallback: use latest daily prediction files to keep the chart dynamic.
        fallback_rows: List[dict] = []
        for pred_file in sorted(PRED_DIR.glob("*_daily_predictions.csv"), reverse=True)[: n_days * 3]:
            preds = pd.read_csv(pred_file)
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
        preds = pd.read_csv(pred_file)
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
        df = pd.read_csv(file)
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

    files = sorted(METRICS_DIR.glob("edge_metrics_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for file in files[:max_files]:
        df = pd.read_csv(file)
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
        df = pd.read_csv(file)
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
        df = pd.read_csv(file)
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
    required_cols = ["strategy", "ending_equity", "return_pct", "win_rate", "max_drawdown"]
    out_file = VIZ_DIR / "strategy_performance.csv"

    if not files:
        pd.DataFrame(columns=required_cols).to_csv(out_file, index=False)
        return pd.DataFrame(columns=required_cols)

    latest = files[0]
    summary = pd.read_csv(latest)
    required = set(required_cols)
    if summary.empty or not required.issubset(summary.columns):
        pd.DataFrame(columns=required_cols).to_csv(out_file, index=False)
        return pd.DataFrame(columns=required_cols)

    out = summary[required_cols].copy()
    out["ending_equity"] = pd.to_numeric(out["ending_equity"], errors="coerce").round(2)
    out["return_pct"] = pd.to_numeric(out["return_pct"], errors="coerce").round(2)
    out["win_rate"] = pd.to_numeric(out["win_rate"], errors="coerce").round(2)
    out["max_drawdown"] = pd.to_numeric(out["max_drawdown"], errors="coerce").round(4)
    out = out.sort_values("return_pct", ascending=False)

    out.to_csv(out_file, index=False)
    logger.info("Saved strategy performance data to %s", out_file)
    return out


def prepare_action_recommendations() -> "pd.DataFrame | None":
    """Prepare per-ticker action recommendations including HOLD option."""
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
    latest_date = preds["predicted_date"].max()
    latest = preds[preds["predicted_date"] == latest_date]

    rows: list[dict] = []
    for _, grp in latest.groupby("ticker"):
        decision = _evaluate_strategies(grp, model_scores, ranked_models, stability_scores)
        score = float(decision["score"])
        if score >= cfg.buy_threshold:
            action = "BUY"
        elif score <= cfg.sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        rows.append(
            {
                "date": pd.Timestamp(latest_date).date().isoformat(),
                "ticker": decision["ticker"],
                "best_model": decision["best_model"],
                "strategy_score": round(score, 4),
                "action": action,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["action", "strategy_score"], ascending=[True, False])
    out_file = VIZ_DIR / "action_recommendations.csv"
    out.to_csv(out_file, index=False)
    logger.info("Saved action recommendations data to %s", out_file)
    return out


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

    (pivot * 100).to_csv(VIZ_DIR / "direction_accuracy.csv")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Accuracy direccional (%) por ticker y modelo", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)")
    fig.tight_layout(pad=2)
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
    _write_viz_manifest(
        [
            pred_df,
            edge_metrics_df,
            feature_stability_df,
            coverage_df,
            candle_df,
            strategy_df,
            action_df,
        ]
    )
    _copy_viz_files()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_viz_tables()
