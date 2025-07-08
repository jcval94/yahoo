"""Utility functions for robust feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay


def _us_election_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return U.S. federal election days between start and end inclusive."""
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    years = range(start.year, end.year + 1)
    days = []
    for year in years:
        if year % 2 != 0:
            continue
        nov_first = pd.Timestamp(year=year, month=11, day=1)
        first_monday_delta = (0 - nov_first.weekday()) % 7
        first_monday = nov_first + pd.DateOffset(days=first_monday_delta)
        election_day = first_monday + pd.Timedelta(days=1)
        if start <= election_day <= end:
            days.append(election_day)
    return pd.DatetimeIndex(days)


def _is_month_end(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean indicator for month-end dates."""
    return dates.is_month_end


def _is_quarter_end(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean indicator for quarter-end dates."""
    return dates.is_quarter_end



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
        # Additional indicators
        df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False, min_periods=1).mean()
        df[f"norm_band_{w}"] = (df["Close"] - df[f"ma_{w}"]) / df[f"std_{w}"]

        if w in [5, 10, 20]:
            df[f"skew_{w}"] = roll.skew()
            df[f"kurt_{w}"] = roll.kurt()

    df["std_ratio_5_20"] = df["std_5"] / df["std_20"]

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
    """Add returns and simple volatility estimates."""
    df = df.copy()
    df["log_return"] = np.log(df["Close"]).diff()
    df["simple_return"] = df["Close"].pct_change()
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
    end = (df.index.max() + pd.offsets.BDay(1)).normalize()
    holidays = cal.holidays(start=df.index.min(), end=end)
    df["is_holiday"] = df.index.normalize().isin(holidays)
    next_days = (df.index + pd.offsets.BDay(1)).normalize()
    df["next_is_holiday"] = next_days.isin(holidays)
    df["prev_is_holiday"] = (
        (df.index.normalize() - pd.Timedelta(days=1)).isin(holidays)
    )

    elections = _us_election_days(df.index.min(), df.index.max())
    df["is_election_day"] = df.index.normalize().isin(elections)
    df["next_is_election_day"] = (
        (df.index.normalize() + BDay()).isin(elections)
    )

    df["is_month_end"] = _is_month_end(df.index)
    df["is_quarter_end"] = _is_quarter_end(df.index)

    return df


def _add_trend_line(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling linear trend estimates for multiple windows."""
    df = df.copy()

    def linfit(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return np.nan
        x = np.arange(len(arr))
        m, b = np.polyfit(x, arr, 1)
        return m * (len(arr) - 1) + b

    for w in [30, 60, 90]:
        df[f"trend_line_{w}"] = (
            df["Close"].rolling(window=w, min_periods=2).apply(linfit, raw=True)
        )

    return df


def _add_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Add STL decomposition components for several rolling windows."""
    df = df.copy()
    try:
        from statsmodels.tsa.seasonal import STL

        for w in [30, 60, 90]:
            trends = []
            seasonals = []
            resids = []
            roll = df["Close"].rolling(window=w, min_periods=2)
            for arr in roll:
                if len(arr) < 2 or len(arr) <= 7:
                    trends.append(np.nan)
                    seasonals.append(np.nan)
                    resids.append(np.nan)
                    continue
                try:
                    result = STL(arr, period=7, robust=True).fit()
                    trends.append(result.trend.iloc[-1])
                    seasonals.append(result.seasonal.iloc[-1])
                    resids.append(result.resid.iloc[-1])
                except Exception:
                    trends.append(np.nan)
                    seasonals.append(np.nan)
                    resids.append(np.nan)

            df[f"stl_trend_{w}"] = trends
            df[f"stl_seasonal_{w}"] = seasonals
            df[f"stl_resid_{w}"] = resids
    except Exception:
        # Statsmodels may not be installed or STL may fail
        pass
    return df


def _add_diff_sign_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary indicators for positive day-over-day changes."""
    df = df.copy()

    columns = ["Close"] + [f"median_{w}" for w in [5, 10, 20, 50]]
    for col in columns:
        if col in df.columns:
            # Use previous value only to avoid look-ahead bias
            diff = df[col] - df[col].shift(1)
            df[f"{col}_up"] = (diff > 0).astype(int)
    return df


def _add_complexity_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a simple entropy-based complexity estimate."""
    df = df.copy()
    sign = (df["Close"].diff() > 0).astype(int)

    def entropy(arr: np.ndarray) -> float:
        counts = np.bincount(arr.astype(int), minlength=2)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    df["entropy_20"] = sign.rolling(window=20, min_periods=1).apply(entropy, raw=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a set of technical indicators to a DataFrame.

    The input series must be sorted in ascending date order. This function
    enforces that ordering before computing indicators. If a ``Ticker`` column
    is present the indicators are computed independently for each ticker.
    """
    if df.empty:
        return df

    def enrich(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_index()
        group = _add_basic_indicators(group)
        group = _add_advanced_indicators(group)
        group = _add_return_features(group)
        group = _add_lag_features(group)
        group = _add_window_stats(group)
        group = _add_diff_sign_features(group)
        group = _add_complexity_feature(group)
        group = _add_seasonal_features(group)
        group = _add_trend_line(group)
        group = _add_decomposition(group)
        return group

    if "Ticker" in df.columns:
        return df.groupby("Ticker", group_keys=False).apply(enrich)
    return enrich(df)
