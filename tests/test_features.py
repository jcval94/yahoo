import pytest
pd = pytest.importorskip("pandas")
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
    assert result.loc[idx[1], "close_lag_1"] == df.loc[idx[0], "Close"]
    assert result.loc[idx[7], "close_lag_7"] == df.loc[idx[0], "Close"]
    expected_sma13 = df["Close"].rolling(window=13, min_periods=1).mean().iloc[12]
    assert result["sma_13"].iloc[12] == expected_sma13

