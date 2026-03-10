"""Utility functions for robust feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta
from pandas.tseries.holiday import USFederalHolidayCalendar
import logging
from .utils import log_df_details

logger = logging.getLogger(__name__)


def get_feature_recalc_rows(safety_rows: int = 180) -> int:
    """Return a safe recomputation tail for rolling/lag features.

    ``safety_rows`` can be increased from the CLI/config to accommodate future
    wider windows without changing this module.
    """
    feature_windows = [
        5,
        7,
        10,
        13,
        14,
        20,
        26,
        30,
        50,
        60,
        90,
    ]
    min_required = max(feature_windows) + 1
    return max(int(safety_rows), min_required)


def _us_election_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return US election days between ``start`` and ``end`` inclusive."""
    years = range(start.year, end.year + 1)
    days = []
    for year in years:
        if year % 2 == 0:
            first_monday = pd.Timestamp(year=year, month=11, day=1) + pd.offsets.Week(weekday=0)
            election = first_monday + pd.Timedelta(days=1)
            days.append(election)
    return pd.DatetimeIndex(days)



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
    prev_close = df["Close"].shift(1)
    df["gap_pct"] = (df["Open"] - prev_close) / prev_close
    df["overnight_return"] = df["gap_pct"]
    df["open_to_close_return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["intraday_return"] = df["open_to_close_return"]
    df["gap_abs"] = df["Open"] - prev_close
    atr_safe = df.get("atr", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    df["gap_atr_norm"] = df["gap_abs"] / atr_safe
    df["drawdown_from_prev_close"] = (df["Low"] - prev_close) / prev_close
    df["log_return"] = np.log(df["Close"]).diff()
    df["simple_return"] = df["Close"].pct_change()
    df["volatility_5"] = df["log_return"].rolling(window=5, min_periods=1).std()
    df["volatility_10"] = df["log_return"].rolling(window=10, min_periods=1).std()
    return df


def _session_bucket(ts: pd.Timestamp) -> str:
    """Map a timestamp to an intraday session bucket."""
    ts = pd.Timestamp(ts)
    minute = ts.hour * 60 + ts.minute
    if 570 <= minute < 690:  # 9:30-11:30
        return "open"
    if 690 <= minute < 900:  # 11:30-15:00
        return "midday"
    if 900 <= minute <= 960:  # 15:00-16:00
        return "close"
    return "close"


def _add_intraday_bucket_features(df: pd.DataFrame, rolling_days: int = 5) -> pd.DataFrame:
    """Add intraday bucket labels and predictive rolling stats by bucket."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    has_intraday_resolution = (df.index.hour != 0).any() or (df.index.minute != 0).any()
    if not has_intraday_resolution:
        df["session_bucket"] = pd.Series(pd.NA, index=df.index, dtype="string")
        for col in [
            "bucket_return_mean_hist",
            "bucket_return_vol_hist",
            "bucket_volume_mean_hist",
            f"bucket_return_mean_{rolling_days}d",
            f"bucket_return_vol_{rolling_days}d",
        ]:
            df[col] = np.nan
        return df

    df["session_bucket"] = pd.Series(df.index.map(_session_bucket), index=df.index, dtype="string")
    bucket_groups = df.groupby("session_bucket", dropna=False)

    df["bucket_return_mean_hist"] = bucket_groups["intraday_return"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["bucket_return_vol_hist"] = bucket_groups["intraday_return"].transform(
        lambda s: s.shift(1).expanding(min_periods=2).std()
    )
    df["bucket_volume_mean_hist"] = bucket_groups["Volume"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    df["trade_date"] = df.index.normalize()
    by_day_bucket = (
        df.groupby(["trade_date", "session_bucket"], dropna=False)
        .agg(day_bucket_return_mean=("intraday_return", "mean"))
        .sort_index()
    )
    bucket_level = by_day_bucket.groupby(level="session_bucket", dropna=False)["day_bucket_return_mean"]
    by_day_bucket[f"bucket_return_mean_{rolling_days}d"] = bucket_level.transform(
        lambda s: s.shift(1).rolling(window=rolling_days, min_periods=1).mean()
    )
    by_day_bucket[f"bucket_return_vol_{rolling_days}d"] = bucket_level.transform(
        lambda s: s.shift(1).rolling(window=rolling_days, min_periods=2).std()
    )

    df = df.join(
        by_day_bucket[[f"bucket_return_mean_{rolling_days}d", f"bucket_return_vol_{rolling_days}d"]],
        on=["trade_date", "session_bucket"],
    )
    df = df.drop(columns=["trade_date"])
    return df


def _recovery_bars_from_drawdown(
    drawdown: pd.Series,
    close: pd.Series,
    *,
    threshold: float,
) -> pd.Series:
    """Return last observed recovery time in bars for a drawdown threshold.

    The metric is updated only when a recovery is fully observed at the
    current bar, so each value can be computed using information available up
    to that timestamp (no look-ahead bias).
    """

    active_events: list[tuple[int, float]] = []
    last_completed = np.nan
    values: list[float] = []

    for i, (dd_value, close_value) in enumerate(zip(drawdown.to_numpy(), close.to_numpy())):
        if pd.notna(dd_value) and dd_value <= -threshold and i > 0:
            active_events.append((i, close.to_numpy()[i - 1]))

        remaining: list[tuple[int, float]] = []
        for start_idx, recovery_level in active_events:
            if pd.notna(close_value) and close_value >= recovery_level:
                last_completed = float(i - start_idx)
            else:
                remaining.append((start_idx, recovery_level))
        active_events = remaining
        values.append(last_completed)

    return pd.Series(values, index=drawdown.index)


def _add_recovery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add recovery-time features for multiple drawdown thresholds."""
    df = df.copy()
    for threshold in (0.05, 0.10, 0.20):
        name = f"recovery_bars_{int(threshold * 100)}pct"
        df[name] = _recovery_bars_from_drawdown(
            df["drawdown_from_prev_close"], df["Close"], threshold=threshold
        )
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

    df["is_month_end"] = df.index.is_month_end
    df["is_month_start"] = df.index.is_month_start
    df["is_monday"] = df.index.dayofweek == 0
    df["is_friday"] = df.index.dayofweek == 4
    df["is_september"] = df.index.month == 9
    dom = df.index.day
    next_bday = df.index + pd.offsets.BDay(1)
    df["is_turn_of_month"] = (dom >= 28) | (next_bday.day <= 3)
    quarter_month = df.index.month.isin([3, 6, 9, 12])
    third_friday = quarter_month & (df.index.dayofweek == 4) & (df.index.day >= 15) & (df.index.day <= 21)
    witching_weeks = df.index.to_period("W-FRI").isin(df.index[third_friday].to_period("W-FRI"))
    df["is_quadruple_witching_week"] = witching_weeks


    cal = USFederalHolidayCalendar()
    end = (df.index.max() + pd.offsets.BDay(1)).normalize()
    holidays = cal.holidays(start=df.index.min(), end=end)
    df["is_holiday"] = df.index.normalize().isin(holidays)
    next_days = (df.index + pd.offsets.BDay(1)).normalize()
    df["next_is_holiday"] = next_days.isin(holidays)
    prev_days = (df.index - pd.Timedelta(days=1)).normalize()
    df["prev_is_holiday"] = prev_days.isin(holidays)

    elections = _us_election_days(df.index.min(), df.index.max())
    df["is_election_day"] = df.index.normalize().isin(elections)
    df["next_is_election_day"] = next_days.isin(elections)

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
        group = _add_recovery_features(group)
        group = _add_lag_features(group)
        group = _add_window_stats(group)
        group = _add_diff_sign_features(group)
        group = _add_complexity_feature(group)
        group = _add_seasonal_features(group)
        group = _add_intraday_bucket_features(group)
        group = _add_trend_line(group)
        group = _add_decomposition(group)
        return group

    if "Ticker" in df.columns:
        groups = [enrich(g) for _, g in df.groupby("Ticker", group_keys=False)]
        result = pd.concat(groups)
    else:
        result = enrich(df)
    log_df_details("technical indicators", result)
    return result
