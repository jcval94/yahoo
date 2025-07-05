import pytest
pd = pytest.importorskip("pandas")
from src.utils.schema_guard import hash_schema, validate_schema


def test_validate_schema_success():
    df = pd.DataFrame({"feat_0": [1], "feat_1": [2], "feat_2": [3]})
    schema_hash = hash_schema(df)
    validate_schema(list(df.columns), df, schema_hash)


def test_validate_schema_missing():
    df = pd.DataFrame({"feat_0": [1], "feat_1": [2]})
    feature_list = ["feat_0", "feat_1", "feat_2"]
    schema_hash = hash_schema(pd.DataFrame(columns=feature_list))
    with pytest.raises(SystemExit):
        validate_schema(feature_list, df, schema_hash)
