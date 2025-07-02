import importlib.util
import pathlib
import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

if not hasattr(pd.DataFrame, 'corr'):
    pytest.skip("pandas not installed", allow_module_level=True)

VS_PATH = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'variable_selection.py'
spec = importlib.util.spec_from_file_location('variable_selection', VS_PATH)
vs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vs)
select_features = vs.select_features


def test_select_features_importance_and_multicollinearity():
    n = 120
    rng = np.random.default_rng(0)
    x1 = rng.random(n)
    x2 = x1 + rng.normal(scale=0.01, size=n)
    x3 = rng.random(n)
    noise = rng.random((n, 15))
    df = pd.DataFrame(noise, columns=[f"z{i}" for i in range(noise.shape[1])])
    df["x1"] = x1
    df["x2"] = x2
    df["x3"] = x3
    df["target"] = 0.7 * x1 + 0.3 * x3 + rng.normal(scale=0.05, size=n)
    selected = select_features(df, "target", sample_size=n, n_splits=3, corr_threshold=0.8)
    assert "x2" not in selected
    assert "x1" in selected
    assert "x3" in selected
    max_features = int(np.sqrt(df.shape[1] - 1)) + 7
    assert len(selected) <= max_features
