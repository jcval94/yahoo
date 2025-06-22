"""ETF selection utilities.

This module proporciona una manera sencilla de elegir hasta diez ETFs
disponibles en el Sistema Internacional de Cotizaciones mexicano. La
selección se basa en la **actividad de los últimos seis meses**, medida
como el volumen promedio de negociación. Además se exige que cada fondo
contenga ciertas acciones claves como Google (GOOGL) y Meta (META). Si
no hay conexión a internet se utiliza una cartera de ejemplo. Para cada
ETF elegido también se intenta incluir la "contraparte" con la
correlación más negativa entre los candidatos para mejorar la
diversificación.
"""
import logging
import socket
from datetime import datetime
from typing import Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _internet_ok(host: str = "query1.finance.yahoo.com", port: int = 443, timeout: int = 3) -> bool:
    """Return True if the host is reachable."""
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        return False


def _fetch_holdings(etf: str) -> List[str]:
    """Return list of holdings symbols for an ETF."""
    if not _internet_ok():
        logger.warning("Runner sin internet. Usando cartera simulada para %s", etf)
        return ["GOOGL", "META", "AAPL"]
    try:
        ticker = yf.Ticker(etf)
        holdings = []
        try:
            hf = ticker.fund_holdings
            if hf is not None and not hf.empty:
                if "symbol" in hf.columns:
                    holdings = hf["symbol"].tolist()
                elif "holding" in hf.columns:
                    holdings = hf["holding"].tolist()
        except Exception:
            info = ticker.info
            holdings = info.get("holdings", [])
            if isinstance(holdings, list):
                holdings = [h.get("symbol") if isinstance(h, dict) else h for h in holdings]
        return holdings
    except Exception:
        logger.error("Failed to get holdings for %s", etf)
        return []


def _find_counterparts(closes: pd.DataFrame) -> Dict[str, str]:
    """Return ETF counterpart with the most negative correlation."""
    returns = closes.pct_change().dropna(how="all")
    corr = returns.corr()
    pairs: Dict[str, str] = {}
    for col in corr.columns:
        ctr = corr[col].idxmin()
        if ctr is not None:
            pairs[col] = ctr
    return pairs


def select_etfs(
    candidates: List[str],
    required: List[str] | None = None,
    end_date: str | None = None,
    limit: int = 10,
) -> List[str]:
    """Return up to ``limit`` ETFs containing ``required`` stocks."""
    # The ranking uses the average trading volume over the last six months
    # to prioritize the most active funds.
    if required is None:
        required = ["GOOGL", "META"]
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    end_dt = pd.to_datetime(end_date)
    start_dt = end_dt - pd.DateOffset(months=6)

    metrics = []
    closes = {}
    for etf in candidates:
        hist = yf.download(
            etf,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False,
        )
        if hist.empty:
            logger.warning("No data for %s", etf)
            continue
        holdings = _fetch_holdings(etf)
        if not all(sym in holdings for sym in required):
            continue
        avg_vol = hist["Volume"].mean()
        metrics.append((etf, avg_vol))
        closes[etf] = hist["Close"]

    metrics.sort(key=lambda x: x[1], reverse=True)
    counter = _find_counterparts(pd.DataFrame(closes)) if closes else {}

    selected: List[str] = []
    for etf, _ in metrics:
        if etf not in selected:
            selected.append(etf)
        cp = counter.get(etf)
        if cp and cp not in selected:
            selected.append(cp)
        if len(selected) >= limit:
            break

    selected = selected[:limit]
    logger.info("Selected ETFs: %s", selected)
    return selected


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(CONFIG_PATH) as cfg_file:
        config = yaml.safe_load(cfg_file)
    etfs = config.get("etfs", [])
    select_etfs(etfs)
