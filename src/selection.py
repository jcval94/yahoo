"""Stock selection utilities."""
import logging
from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for a ticker."""
    # Explicitly disable auto_adjust to keep raw prices consistent across
    # yfinance versions (the default switched to True in 0.2.37).
    return yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )


def select_tickers(candidates: List[str], end_date: str) -> List[str]:
    """Select 10 tickers based on volume, stability and performance."""
    end_dt = pd.to_datetime(end_date)
    start_dt = end_dt - pd.DateOffset(months=6)
    metrics = {}

    for ticker in candidates:
        df = _fetch_history(ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        if df.empty:
            logger.warning("No data for %s", ticker)
            continue
        avg_vol = df["Volume"].mean()
        returns = df["Close"].pct_change().dropna()
        volatility = returns.std()
        total_return = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
        metrics[ticker] = {
            "volume": avg_vol,
            "volatility": volatility,
            "return": total_return,
        }

    vol_sorted = sorted(metrics.items(), key=lambda x: x[1]["volume"], reverse=True)
    top_volume = [t for t, _ in vol_sorted[:5]]

    stable_sorted = sorted(metrics.items(), key=lambda x: x[1]["volatility"])
    most_stable = [t for t, _ in stable_sorted if t not in top_volume][:3]

    loser_sorted = sorted(metrics.items(), key=lambda x: x[1]["return"])
    biggest_losers = [t for t, _ in loser_sorted if t not in top_volume + most_stable][:2]

    selection = top_volume + most_stable + biggest_losers
    logger.info("Selected tickers: %s", selection)
    return selection


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(CONFIG_PATH) as cfg_file:
        config = yaml.safe_load(cfg_file)

    tickers = config.get("etfs", [])
    today = datetime.today().strftime("%Y-%m-%d")
    selected = select_tickers(tickers, today)
    print("Selected tickers:", ", ".join(selected))
