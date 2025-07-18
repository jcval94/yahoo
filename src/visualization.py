import logging
from pathlib import Path
from typing import Dict, List

try:  # Optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - pandas may not be installed
    pd = None  # type: ignore

from base64 import b64decode
def load_config(_: Path) -> dict:
    """Fallback loader when dependencies are missing."""
    return {}


def log_df_details(name: str, df, head: int = 3) -> None:
    """Minimal logger replacement."""
    logger.info("%s generated", name)

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
CONFIG = load_config(CONFIG_PATH)

DATA_DIR = Path(__file__).resolve().parents[1] / CONFIG.get("data_dir", "data")
PRED_DIR = Path(__file__).resolve().parents[1] / "results" / "predicts"
FEATURE_DIR = Path(__file__).resolve().parents[1] / "results" / "features"
VIZ_DIR = Path(__file__).resolve().parents[1] / "results" / "viz"
VIZ_DIR.mkdir(exist_ok=True, parents=True)

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
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_file = VIZ_DIR / "candlestick.csv"
    result.to_csv(out_file, index=False)
    logger.info("Saved candlestick data to %s", out_file)
    log_df_details("candlestick data", result)
    return result


def prepare_pred_vs_real() -> "pd.DataFrame | None":
    """Return table comparing predictions with actual prices."""
    if pd is None:
        return None
    pred_file = _latest_csv(PRED_DIR, "*_edge_prediction.csv")
    if pred_file is None:
        logger.info("No edge prediction file found")
        return pd.DataFrame()
    preds = pd.read_csv(pred_file)
    if preds.empty:
        return pd.DataFrame()
    predict_date = pd.to_datetime(preds["Predicted"].iloc[0]).date()
    rows: List[Dict[str, float]] = []
    for ticker, grp in preds.groupby("ticker"):
        data_path = DATA_DIR / f"{ticker}.csv"
        if not data_path.exists():
            continue
        df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
        idx = df.index.date == predict_date
        if not idx.any():
            continue
        actual = df.loc[idx, "Close"].iloc[0]
        for _, r in grp.iterrows():
            rows.append(
                {
                    "ticker": ticker,
                    "model": r["model"],
                    "pred": float(r["pred"]),
                    "real": float(actual),
                    "Predicted": r["Predicted"],
                }
            )
    result = pd.DataFrame(rows)
    out_file = VIZ_DIR / "pred_vs_real.csv"
    result.to_csv(out_file, index=False)
    logger.info("Saved prediction comparison to %s", out_file)
    log_df_details("pred vs real", result)
    return result


def prepare_best_variables(top_n: int = 10) -> "pd.DataFrame | None":
    """Return the top features ranked by the average of ``importance_mean``."""
    if pd is None:
        return None
    feat_file = _latest_csv(FEATURE_DIR, "features_*_*.csv")
    if feat_file is None:
        logger.info("No feature importance file found")
        return pd.DataFrame()
    df = pd.read_csv(feat_file)
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby("feature")["importance_mean"]
        .mean()
        .reset_index(name="importance_mean")
    )
    top_df = agg.sort_values("importance_mean", ascending=False).head(top_n)
    out_file = VIZ_DIR / "best_variables.csv"
    top_df.to_csv(out_file, index=False)
    logger.info("Saved best variables to %s", out_file)
    log_df_details("best variables", top_df)
    return top_df


def _plot_candlestick(df: "pd.DataFrame | None", out_file: Path) -> None:
    """Create line plot of closing prices."""
    if pd is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "Candlestick")
        return
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    pivot = df.pivot(index="Date", columns="ticker", values="Close")
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(ax=ax, linewidth=1.5)
    ax.set_title("Precio de cierre - últimos 15 días", fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Cierre")
    ax.grid(True, linestyle="--", alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=len(labels), frameon=False)
    fig.tight_layout(pad=2)
    fig.savefig(out_file)
    fig.savefig(out_file.with_suffix(".svg"))
    plt.close(fig)


def _plot_pred_vs_real(df: "pd.DataFrame | None", out_file: Path) -> None:
    """Scatter plot of predicted vs real prices."""
    if pd is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "Pred vs Real")
        return
    import matplotlib.pyplot as plt

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
    if pd is None or df is None or getattr(df, "empty", True):
        _write_placeholder(out_file, "Best Variables")
        return
    import matplotlib.pyplot as plt

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


def create_viz_tables() -> None:
    """Generate all visualization tables."""
    candle_df = prepare_candlestick_data()
    pred_df = prepare_pred_vs_real()
    best_df = prepare_best_variables()

    _plot_candlestick(candle_df, VIZ_DIR / "candlestick.png")
    _plot_pred_vs_real(pred_df, VIZ_DIR / "pred_vs_real.png")
    _plot_best_variables(best_df, VIZ_DIR / "best_variables.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_viz_tables()
