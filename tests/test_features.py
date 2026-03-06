import pytest
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("ta")
from src.features import add_technical_indicators


def test_lag_features_present():
    idx = pd.date_range(start="2020-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "Open": range(30),
        "High": range(30),
        "Low": range(30),
        "Close": range(30),
        "Adj Close": range(30),
        "Volume": range(30)
    }, index=idx)
    result = add_technical_indicators(df)
    for col in ["close_lag_1", "close_lag_7", "close_lag_14", "sma_13", "sma_26"]:
        assert col in result.columns
    for col in ["log_return", "volatility_5", "volatility_10"]:
        assert col in result.columns
    assert result.loc[idx[1], "close_lag_1"] == df.loc[idx[0], "Close"]
    expected_lr = (np.log(df["Close"]).diff().iloc[1])
    assert result["log_return"].iloc[1] == expected_lr
    assert result.loc[idx[7], "close_lag_7"] == df.loc[idx[0], "Close"]
    expected_sma13 = df["Close"].rolling(window=13, min_periods=1).mean().iloc[12]
    assert result["sma_13"].iloc[12] == expected_sma13


def test_trend_line_columns_added():
    idx = pd.date_range(start="2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "Open": range(100),
        "High": range(100),
        "Low": range(100),
        "Close": range(100),
        "Adj Close": range(100),
        "Volume": range(100)
    }, index=idx)
    result = add_technical_indicators(df)
    for w in [30, 60, 90]:
        assert f"trend_line_{w}" in result.columns


def test_stl_columns_added():
    pytest.importorskip("statsmodels")
    idx = pd.date_range(start="2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "Open": range(100),
        "High": range(100),
        "Low": range(100),
        "Close": range(100),
        "Adj Close": range(100),
        "Volume": range(100)
    }, index=idx)
    result = add_technical_indicators(df)
    for w in [30, 60, 90]:
        for comp in ["trend", "seasonal", "resid"]:
            assert f"stl_{comp}_{w}" in result.columns


def test_unsorted_input_is_sorted_before_features():
    idx = pd.date_range(start="2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Close": range(10)}, index=idx)
    df_desc = df.iloc[::-1]
    result = add_technical_indicators(df_desc)
    expected = add_technical_indicators(df)
    pd.testing.assert_frame_equal(result, expected)


def test_diff_sign_features():
    idx = pd.date_range(start="2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({
        "Open": [1, 2, 1],
        "High": [1, 2, 1],
        "Low": [1, 2, 1],
        "Close": [1, 2, 1],
        "Adj Close": [1, 2, 1],
        "Volume": [1, 1, 1],
    }, index=idx)
    result = add_technical_indicators(df)
    assert "Close_up" in result.columns
    expected_close = (df["Close"].diff() > 0).astype(int)
    pd.testing.assert_series_equal(result["Close_up"], expected_close, check_names=False)

    for w in [5, 10, 20, 50]:
        col = f"median_{w}_up"
        assert col in result.columns
        expected = (result[f"median_{w}"].diff() > 0).astype(int)
        pd.testing.assert_series_equal(result[col], expected, check_names=False)


def test_ema_and_norm_band_and_simple_return():
    idx = pd.date_range(start="2020-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "Open": range(60),
        "High": range(60),
        "Low": range(60),
        "Close": range(60),
        "Adj Close": range(60),
        "Volume": range(60),
    }, index=idx)
    result = add_technical_indicators(df)
    for w in [5, 10, 20, 50]:
        assert f"ema_{w}" in result.columns
        assert f"norm_band_{w}" in result.columns

    assert "simple_return" in result.columns
    expected_simple = df["Close"].pct_change()
    pd.testing.assert_series_equal(result["simple_return"], expected_simple, check_names=False)


def test_std_ratio_and_moments_and_entropy():
    idx = pd.date_range(start="2020-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "Open": range(40),
        "High": range(40),
        "Low": range(40),
        "Close": range(40),
        "Adj Close": range(40),
        "Volume": range(40),
    }, index=idx)
    result = add_technical_indicators(df)

    # std ratio
    assert "std_ratio_5_20" in result.columns
    expected_ratio = result["std_5"] / result["std_20"]
    pd.testing.assert_series_equal(result["std_ratio_5_20"], expected_ratio, check_names=False)

    # skew and kurtosis
    for w in [5, 10, 20]:
        assert f"skew_{w}" in result.columns
        assert f"kurt_{w}" in result.columns
        expected_skew = df["Close"].rolling(window=w, min_periods=1).skew()
        expected_kurt = df["Close"].rolling(window=w, min_periods=1).kurt()
        pd.testing.assert_series_equal(result[f"skew_{w}"], expected_skew, check_names=False)
        pd.testing.assert_series_equal(result[f"kurt_{w}"], expected_kurt, check_names=False)

    # entropy complexity
    assert "entropy_20" in result.columns
    sign = (df["Close"].diff() > 0).astype(int)

    def entropy(arr):
        counts = np.bincount(arr.astype(int), minlength=2)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -(probs * np.log(probs)).sum()

    expected_entropy = sign.rolling(window=20, min_periods=1).apply(entropy, raw=True)
    pd.testing.assert_series_equal(result["entropy_20"], expected_entropy, check_names=False)


def test_next_is_holiday():
    idx = pd.to_datetime([
        "2020-12-23",
        "2020-12-24",
        "2020-12-28",
        "2020-12-29",
    ])
    df = pd.DataFrame(
        {
            "Open": range(len(idx)),
            "High": range(len(idx)),
            "Low": range(len(idx)),
            "Close": range(len(idx)),
            "Adj Close": range(len(idx)),
            "Volume": range(len(idx)),
        },
        index=idx,
    )
    result = add_technical_indicators(df)
    cal = pd.tseries.holiday.USFederalHolidayCalendar()
    end = (idx.max() + pd.offsets.BDay(1)).normalize()
    holidays = cal.holidays(start=idx.min(), end=end)
    expected = (idx + pd.offsets.BDay(1)).normalize().isin(holidays)
    pd.testing.assert_series_equal(
        result["next_is_holiday"],
        pd.Series(expected, index=idx, name="next_is_holiday"),
    )


def test_election_day_columns():
    idx = pd.to_datetime([
        "2020-11-02",
        "2020-11-03",
        "2020-11-04",
        "2022-11-07",
        "2022-11-08",
        "2022-11-09",
    ])
    df = pd.DataFrame(
        {
            "Open": range(len(idx)),
            "High": range(len(idx)),
            "Low": range(len(idx)),
            "Close": range(len(idx)),
            "Adj Close": range(len(idx)),
            "Volume": range(len(idx)),
        },
        index=idx,
    )
    result = add_technical_indicators(df)

    assert result.loc[pd.Timestamp("2020-11-03"), "is_election_day"]
    assert result.loc[pd.Timestamp("2022-11-08"), "is_election_day"]
    assert result.loc[pd.Timestamp("2020-11-02"), "next_is_election_day"]
    assert result.loc[pd.Timestamp("2022-11-07"), "next_is_election_day"]
    assert not result.loc[pd.Timestamp("2020-11-04"), "is_election_day"]


def test_is_month_end_indicator():
    idx = pd.to_datetime([
        "2020-01-30",
        "2020-01-31",
        "2020-02-03",
    ])
    df = pd.DataFrame(
        {
            "Open": range(len(idx)),
            "High": range(len(idx)),
            "Low": range(len(idx)),
            "Close": range(len(idx)),
            "Adj Close": range(len(idx)),
            "Volume": range(len(idx)),
        },
        index=idx,
    )
    result = add_technical_indicators(df)

    assert result.loc[pd.Timestamp("2020-01-31"), "is_month_end"]
    assert not result.loc[pd.Timestamp("2020-01-30"), "is_month_end"]


def test_prev_is_holiday():
    idx = pd.to_datetime([
        "2020-11-25",
        "2020-11-26",
        "2020-11-27",
        "2020-11-30",
    ])
    df = pd.DataFrame(
        {
            "Open": range(len(idx)),
            "High": range(len(idx)),
            "Low": range(len(idx)),
            "Close": range(len(idx)),
            "Adj Close": range(len(idx)),
            "Volume": range(len(idx)),
        },
        index=idx,
    )
    result = add_technical_indicators(df)
    cal = pd.tseries.holiday.USFederalHolidayCalendar()
    end = (idx.max() + pd.offsets.BDay(1)).normalize()
    holidays = cal.holidays(start=idx.min(), end=end)
    expected = (idx - pd.Timedelta(days=1)).normalize().isin(holidays)
    pd.testing.assert_series_equal(
        result["prev_is_holiday"],
        pd.Series(expected, index=idx, name="prev_is_holiday"),
    )



def test_gap_and_intraday_return_columns():
    idx = pd.date_range(start="2021-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 110, 105, 100],
            "High": [102, 112, 107, 101],
            "Low": [99, 108, 100, 96],
            "Close": [100, 109, 104, 98],
            "Adj Close": [100, 109, 104, 98],
            "Volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    result = add_technical_indicators(df)

    prev_close = df["Close"].shift(1)
    expected_gap = (df["Open"] - prev_close) / prev_close
    expected_open_to_close = (df["Close"] - df["Open"]) / df["Open"]
    expected_drawdown = (df["Low"] - prev_close) / prev_close

    pd.testing.assert_series_equal(result["gap_pct"], expected_gap, check_names=False)
    pd.testing.assert_series_equal(
        result["overnight_return"], expected_gap, check_names=False
    )
    pd.testing.assert_series_equal(
        result["open_to_close_return"], expected_open_to_close, check_names=False
    )
    pd.testing.assert_series_equal(
        result["drawdown_from_prev_close"], expected_drawdown, check_names=False
    )


def test_recovery_bars_thresholds_without_lookahead():
    idx = pd.date_range(start="2021-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 100, 99, 95, 97, 100],
            "High": [101, 101, 100, 96, 98, 101],
            "Low": [99, 94, 97, 94, 96, 99],
            "Close": [100, 99, 98, 95, 97, 100],
            "Adj Close": [100, 99, 98, 95, 97, 100],
            "Volume": [1, 1, 1, 1, 1, 1],
        },
        index=idx,
    )

    result = add_technical_indicators(df)

    assert "recovery_bars_5pct" in result.columns
    assert "recovery_bars_10pct" in result.columns
    assert "recovery_bars_20pct" in result.columns

    # Day 2 has a drawdown below -5% from previous close but has not recovered yet.
    assert np.isnan(result.loc[idx[1], "recovery_bars_5pct"])
    # Recovery happens on day 6 (close returns to prior close level), taking 4 bars.
    assert result.loc[idx[5], "recovery_bars_5pct"] == 4.0
    # No >10% or >20% drawdown event in this sample.
    assert result["recovery_bars_10pct"].isna().all()
    assert result["recovery_bars_20pct"].isna().all()


def test_new_return_features_and_gap_normalization_by_atr():
    idx = pd.date_range(start="2021-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 140, num=40),
            "High": np.linspace(101, 141, num=40),
            "Low": np.linspace(99, 139, num=40),
            "Close": np.linspace(100, 140, num=40),
            "Adj Close": np.linspace(100, 140, num=40),
            "Volume": np.ones(40),
        },
        index=idx,
    )
    result = add_technical_indicators(df)

    prev_close = df["Close"].shift(1)
    expected_intraday = (df["Close"] - df["Open"]) / df["Open"]
    expected_gap_abs = df["Open"] - prev_close

    pd.testing.assert_series_equal(result["intraday_return"], expected_intraday, check_names=False)
    pd.testing.assert_series_equal(result["gap_abs"], expected_gap_abs, check_names=False)
    pd.testing.assert_series_equal(
        result["gap_atr_norm"],
        result["gap_abs"] / result["atr"].replace(0, np.nan),
        check_names=False,
    )


def test_calendar_flags_added():
    idx = pd.date_range("2024-09-02", periods=25, freq="B")
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 124, num=len(idx)),
            "High": np.linspace(101, 125, num=len(idx)),
            "Low": np.linspace(99, 123, num=len(idx)),
            "Close": np.linspace(100, 124, num=len(idx)),
            "Adj Close": np.linspace(100, 124, num=len(idx)),
            "Volume": np.ones(len(idx)),
        },
        index=idx,
    )
    result = add_technical_indicators(df)

    assert result.loc[pd.Timestamp("2024-09-16"), "is_monday"]
    assert result.loc[pd.Timestamp("2024-09-20"), "is_friday"]
    assert result.loc[pd.Timestamp("2024-09-16"), "is_september"]
    assert result.loc[pd.Timestamp("2024-10-01"), "is_month_start"]
    assert result.loc[pd.Timestamp("2024-09-30"), "is_turn_of_month"]
    assert result.loc[pd.Timestamp("2024-10-01"), "is_turn_of_month"]
    assert result.loc[pd.Timestamp("2024-09-16"), "is_quadruple_witching_week"]
    assert result.loc[pd.Timestamp("2024-09-20"), "is_quadruple_witching_week"]


def test_intraday_session_bucket_and_rolling_stats_no_leakage():
    idx = pd.to_datetime([
        "2024-01-02 09:30", "2024-01-02 12:00", "2024-01-02 15:30",
        "2024-01-03 09:30", "2024-01-03 12:00", "2024-01-03 15:30",
        "2024-01-04 09:30", "2024-01-04 12:00", "2024-01-04 15:30",
        "2024-01-05 09:30", "2024-01-05 12:00", "2024-01-05 15:30",
        "2024-01-08 09:30", "2024-01-08 12:00", "2024-01-08 15:30",
    ])
    opens = [100, 100, 100, 103, 101, 99, 98, 102, 101, 102, 104, 103, 105, 106, 107]
    closes = [101, 99, 100, 102, 102, 100, 99, 101, 102, 103, 103, 104, 106, 105, 108]
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": [max(o, c) + 1 for o, c in zip(opens, closes)],
            "Low": [min(o, c) - 1 for o, c in zip(opens, closes)],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 900, 950, 980, 990, 995, 1005, 1010, 1020, 1030],
        },
        index=idx,
    )
    result = add_technical_indicators(df)

    assert result.loc[pd.Timestamp("2024-01-02 09:30"), "session_bucket"] == "open"
    assert result.loc[pd.Timestamp("2024-01-02 12:00"), "session_bucket"] == "midday"
    assert result.loc[pd.Timestamp("2024-01-02 15:30"), "session_bucket"] == "close"

    # First observation per bucket has no historical values.
    assert np.isnan(result.loc[pd.Timestamp("2024-01-02 09:30"), "bucket_return_mean_hist"])
    # Second open bar uses only previous open bar value (no look-ahead).
    first_open_intraday = result.loc[pd.Timestamp("2024-01-02 09:30"), "intraday_return"]
    assert result.loc[pd.Timestamp("2024-01-03 09:30"), "bucket_return_mean_hist"] == first_open_intraday

    # Daily rolling by bucket must also be predictive: day-2 open uses only day-1 open block return.
    day1_open_mean = result.loc[pd.Timestamp("2024-01-02 09:30"), "intraday_return"]
    assert result.loc[pd.Timestamp("2024-01-03 09:30"), "bucket_return_mean_5d"] == day1_open_mean
