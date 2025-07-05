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


def _add_window_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling window statistics for multiple horizons."""
    df = df.copy()
    windows = [5, 10, 20, 50]
    for w in windows:
        roll = df["Close"].rolling(window=w, min_periods=1)
        df[f"ma_{w}"] = roll.mean()
        df[f"min_{w}"] = roll.min()
        df[f"q25_{w}"] = roll.quantile(0.25)
        df[f"median_{w}"] = roll.quantile(0.5)
        df[f"q75_{w}"] = roll.quantile(0.75)
        df[f"max_{w}"] = roll.max()
        df[f"std_{w}"] = roll.std()
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


def _add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns and simple volatility estimates."""
    df = df.copy()
    df["log_return"] = np.log(df["Close"]).diff()
    df["volatility_5"] = df["log_return"].rolling(window=5, min_periods=1).std()
    df["volatility_10"] = df["log_return"].rolling(window=10, min_periods=1).std()
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged values and short moving averages of the close price."""
    df = df.copy()
    df["close_lag_1"] = df["Close"].shift(1)
    df["close_lag_7"] = df["Close"].shift(7)
    df["close_lag_14"] = df["Close"].shift(14)
    df["sma_13"] = df["Close"].rolling(window=13, min_periods=1).mean()
    df["sma_26"] = df["Close"].rolling(window=26, min_periods=1).mean()
    return df


def _add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar based features and seasonal cycles."""
    df = df.copy()

    df["dow"] = df.index.dayofweek
    df["dom"] = df.index.day
    df["month"] = df.index.month

    day_of_year = df.index.dayofyear.astype(float)
    df["sin_week"] = np.sin(2 * np.pi * day_of_year / 7)
    df["cos_week"] = np.cos(2 * np.pi * day_of_year / 7)
    df["sin_month"] = np.sin(2 * np.pi * day_of_year / 30.5)
    df["cos_month"] = np.cos(2 * np.pi * day_of_year / 30.5)
    df["sin_quarter"] = np.sin(2 * np.pi * day_of_year / 91.25)
    df["cos_quarter"] = np.cos(2 * np.pi * day_of_year / 91.25)
    df["sin_year"] = np.sin(2 * np.pi * day_of_year / 365)
    df["cos_year"] = np.cos(2 * np.pi * day_of_year / 365)


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
        group = _add_return_features(group)
        group = _add_lag_features(group)
        group = _add_window_stats(group)
        group = _add_seasonal_features(group)
        group = _add_trend_line(group)
        group = _add_decomposition(group)
        return group

    if "Ticker" in df.columns:
        return df.groupby("Ticker", group_keys=False).apply(enrich)
    return enrich(df)
