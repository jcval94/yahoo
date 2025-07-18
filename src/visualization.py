import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils import load_config, log_df_details

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
CONFIG = load_config(CONFIG_PATH)

DATA_DIR = Path(__file__).resolve().parents[1] / CONFIG.get("data_dir", "data")
PRED_DIR = Path(__file__).resolve().parents[1] / "results" / "predicts"
FEATURE_DIR = Path(__file__).resolve().parents[1] / "results" / "features"
VIZ_DIR = Path(__file__).resolve().parents[1] / "results" / "viz"
VIZ_DIR.mkdir(exist_ok=True, parents=True)


def _latest_csv(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def prepare_candlestick_data(n_days: int = 60) -> pd.DataFrame:
    """Return OHLC data for recent days of all tickers."""
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


def prepare_pred_vs_real() -> pd.DataFrame:
    """Return table comparing predictions with actual prices."""
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


def prepare_best_variables(top_n: int = 5) -> pd.DataFrame:
    """Return the top features by importance for each model."""
    feat_file = _latest_csv(FEATURE_DIR, "features_*_*.csv")
    if feat_file is None:
        logger.info("No feature importance file found")
        return pd.DataFrame()
    df = pd.read_csv(feat_file)
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["model", "importance_mean"], ascending=[True, False])
    top_df = df.groupby("model").head(top_n).reset_index(drop=True)
    out_file = VIZ_DIR / "best_variables.csv"
    top_df.to_csv(out_file, index=False)
    logger.info("Saved best variables to %s", out_file)
    log_df_details("best variables", top_df)
    return top_df


def _plot_candlestick(df: pd.DataFrame, out_file: Path) -> None:
    """Create line plot of closing prices."""
    if df.empty:
        return
    import matplotlib.pyplot as plt

    pivot = df.pivot(index="Date", columns="ticker", values="Close")
    pivot.plot(figsize=(10, 4))
    plt.title("Precio de cierre - últimos 60 días")
    plt.xlabel("Fecha")
    plt.ylabel("Cierre")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.savefig(out_file.with_suffix(".svg"))
    plt.close()


def _plot_pred_vs_real(df: pd.DataFrame, out_file: Path) -> None:
    """Scatter plot of predicted vs real prices."""
    if df.empty:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["real"], df["pred"], alpha=0.7)
    min_val = min(df["real"].min(), df["pred"].min())
    max_val = max(df["real"].max(), df["pred"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicción")
    ax.set_title("Predicción vs Real")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.savefig(out_file.with_suffix(".svg"))
    plt.close()


def _plot_best_variables(df: pd.DataFrame, out_file: Path) -> None:
    """Bar plot of feature importance."""
    if df.empty:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(kind="barh", x="feature", y="importance_mean", ax=ax)
    ax.set_xlabel("Importancia")
    ax.set_ylabel("Variable")
    ax.set_title("Variables más importantes")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.savefig(out_file.with_suffix(".svg"))
    plt.close()


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
