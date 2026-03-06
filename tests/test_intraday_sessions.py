import pytest

pd = pytest.importorskip("pandas")

from src.abt.build_abt import add_session_column
from src.training import split_train_test
from src.predict import _next_prediction_timestamp


def test_add_session_column_assigns_expected_labels():
    idx = pd.to_datetime(
        [
            "2024-01-02 08:30:00",
            "2024-01-02 10:00:00",
            "2024-01-02 13:00:00",
            "2024-01-02 15:45:00",
            "2024-01-02 17:30:00",
        ],
        utc=True,
    )
    df = pd.DataFrame({"Close": [1, 2, 3, 4, 5]}, index=idx)

    out = add_session_column(df)

    assert out["session"].tolist() == [
        "after_hours",
        "open",
        "midday",
        "close",
        "after_hours",
    ]


def test_split_train_test_intraday_uses_bar_windows():
    idx = pd.date_range("2024-01-01 09:30", periods=30, freq="5min")
    df = pd.DataFrame({"Close": range(30)}, index=idx)

    train, test = split_train_test(df, frequency="intraday", val_bars=6)

    assert len(train) == 24
    assert len(test) == 6
    assert train.index.max() < test.index.min()


def test_next_prediction_timestamp_intraday_uses_next_bar():
    idx = pd.date_range("2024-01-01 09:30", periods=3, freq="5min")

    pred_ts = _next_prediction_timestamp(idx, "intraday")

    assert pred_ts == idx[-1] + pd.Timedelta(minutes=5)
