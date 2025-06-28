"""Utility functions for robust feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta
from pandas.tseries.holiday import USFederalHolidayCalendar



def _add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["sma_20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()
    return df


def _add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df["Close"])
    df["bb_bbm"] = bb.bollinger_mavg()
    df["bb_bbh"] = bb.bollinger_hband()
    df["bb_bbl"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_bbh"] - df["bb_bbl"]) / df["bb_bbm"]

    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["stoch"] = stoch.stoch()
    df["stoch_signal"] = stoch.stoch_signal()

    atr = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"])
    df["atr"] = atr.average_true_range()

    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    df["obv"] = obv.on_balance_volume()

    return df


def _add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, day-of-month and US holiday flags."""
    df = df.copy()
    df["dow"] = df.index.dayofweek
    df["dom"] = df.index.day

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    df["is_holiday"] = df.index.normalize().isin(holidays)

    return df


def _add_trend_line(df: pd.DataFrame) -> pd.DataFrame:
    """Add a simple linear trend line over the close price."""
    df = df.copy()
    if len(df) < 2:
        df["trend_line"] = df["Close"]
        return df

    x = np.arange(len(df))
    coeffs = np.polyfit(x, df["Close"].values, 1)
    df["trend_line"] = coeffs[0] * x + coeffs[1]
    return df


def _add_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Add STL decomposition components if statsmodels is available."""
    df = df.copy()
    try:
        from statsmodels.tsa.seasonal import STL

        stl = STL(df["Close"], period=7, robust=True)
        result = stl.fit()
        df["stl_trend"] = result.trend
        df["stl_seasonal"] = result.seasonal
        df["stl_resid"] = result.resid
    except Exception:
        # Statsmodels may not be installed or STL may fail
        pass
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a set of technical indicators to a DataFrame.

    If a ``Ticker`` column is present the indicators are computed
    independently for each ticker.
    """
    if df.empty:
        return df

    def enrich(group: pd.DataFrame) -> pd.DataFrame:
        group = _add_basic_indicators(group)
        group = _add_advanced_indicators(group)
        group = _add_seasonal_features(group)
        group = _add_trend_line(group)
        group = _add_decomposition(group)
        return group

    if "Ticker" in df.columns:
        return df.groupby("Ticker", group_keys=False).apply(enrich)
    return enrich(df)
