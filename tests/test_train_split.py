import pytest
pd = pytest.importorskip("pandas")
from src.training import split_train_test


def test_split_no_overlap():
    idx = pd.date_range('2020-01-01', periods=40, freq='D')
    df = pd.DataFrame({'Close': range(40)}, index=idx)
    train, test = split_train_test(df)
    assert not train.index.intersection(test.index).any()
    if not train.empty and not test.empty:
        assert train.index.max() < test.index.min()
