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
    pd.testing.assert_series_equal(result["Close_up"], expected_close, check_name=False)

    for w in [5, 10, 20, 50]:
        col = f"median_{w}_up"
        assert col in result.columns
        expected = (result[f"median_{w}"].diff() > 0).astype(int)
        pd.testing.assert_series_equal(result[col], expected, check_name=False)


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
    pd.testing.assert_series_equal(result["simple_return"], expected_simple, check_name=False)


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
    pd.testing.assert_series_equal(result["std_ratio_5_20"], expected_ratio, check_name=False)

    # skew and kurtosis
    for w in [5, 10, 20]:
        assert f"skew_{w}" in result.columns
        assert f"kurt_{w}" in result.columns
        expected_skew = df["Close"].rolling(window=w, min_periods=1).skew()
        expected_kurt = df["Close"].rolling(window=w, min_periods=1).kurt()
        pd.testing.assert_series_equal(result[f"skew_{w}"], expected_skew, check_name=False)
        pd.testing.assert_series_equal(result[f"kurt_{w}"], expected_kurt, check_name=False)

    # entropy complexity
    assert "entropy_20" in result.columns
    sign = (df["Close"].diff() > 0).astype(int)

    def entropy(arr):
        counts = np.bincount(arr.astype(int), minlength=2)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -(probs * np.log(probs)).sum()

    expected_entropy = sign.rolling(window=20, min_periods=1).apply(entropy, raw=True)
    pd.testing.assert_series_equal(result["entropy_20"], expected_entropy, check_name=False)


def test_election_day_and_month_end_features():
    idx = pd.DatetimeIndex([
        "2020-11-02",
        "2020-11-03",
        "2022-11-07",
        "2022-11-08",
    ])
    df = pd.DataFrame({
        "Open": range(len(idx)),
        "High": range(len(idx)),
        "Low": range(len(idx)),
        "Close": range(len(idx)),
        "Adj Close": range(len(idx)),
        "Volume": range(len(idx)),
    }, index=idx)
    result = add_technical_indicators(df)
    assert "is_election_day" in result.columns
    assert "next_is_election_day" in result.columns
    assert result.loc[pd.Timestamp("2020-11-03"), "is_election_day"]
    assert result.loc[pd.Timestamp("2020-11-02"), "next_is_election_day"]
    assert result.loc[pd.Timestamp("2022-11-08"), "is_election_day"]
    assert result.loc[pd.Timestamp("2022-11-07"), "next_is_election_day"]

    idx2 = pd.date_range("2020-01-29", periods=5, freq="D")
    df2 = pd.DataFrame({
        "Open": range(5),
        "High": range(5),
        "Low": range(5),
        "Close": range(5),
        "Adj Close": range(5),
        "Volume": range(5),
    }, index=idx2)
    result2 = add_technical_indicators(df2)
    assert "is_month_end" in result2.columns
    assert result2.loc[pd.Timestamp("2020-01-31"), "is_month_end"]

