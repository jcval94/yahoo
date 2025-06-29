import importlib.util
import pathlib
import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

if not hasattr(pd.DataFrame, 'corr'):
    pytest.skip("pandas not installed", allow_module_level=True)

VS_PATH = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'variable_selection.py'
spec = importlib.util.spec_from_file_location('variable_selection', VS_PATH)
vs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vs)
select_features = vs.select_features


def test_select_features_multicollinearity():
    n = 100
    x1 = np.linspace(0, 1, n)
    x2 = [v * 0.95 + 0.05 for v in x1]
    x3 = np.linspace(1, 0, n)
    y = [0.5 * x1[i] + 0.2 * x3[i] for i in range(n)]
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'target': y})
    selected = select_features(df, 'target', corr_threshold=0.8, relevance_threshold=0.1)
    assert 'x2' not in selected
    assert 'x1' in selected
    assert 'x3' in selected
