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


def create_viz_tables() -> None:
    """Generate all visualization tables."""
    prepare_candlestick_data()
    prepare_pred_vs_real()
    prepare_best_variables()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_viz_tables()
