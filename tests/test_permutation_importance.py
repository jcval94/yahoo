import importlib.util
import pathlib
import sys
import types
import pytest

# Provide minimal pandas stub if not installed
_pd_stub = None
if 'pandas' not in sys.modules:
    pd_stub = types.ModuleType('pandas')
    class DataFrame(list):
        def __init__(self, data):
            self.data = data
            self.columns = list(data.keys())
        def __getitem__(self, key):
            return self.data[key]
    class Series(list):
        pass
    pd_stub.DataFrame = DataFrame
    pd_stub.Series = Series
    sys.modules['pandas'] = pd_stub
    _pd_stub = pd_stub

MODULE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'permutation_importance.py'
spec = importlib.util.spec_from_file_location('permutation_importance', MODULE)
perm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perm)
compute_permutation_importance = perm.compute_permutation_importance

if _pd_stub is not None:
    del sys.modules['pandas']


@pytest.mark.skipif('sklearn' not in pytest.importorskip('sklearn').__dict__, reason='sklearn not available')
def test_compute_permutation_importance_basic():
    pd = pytest.importorskip('pandas')
    from sklearn.linear_model import LinearRegression

    X = pd.DataFrame({'a': [0, 1, 2, 3], 'b': [1, 1, 1, 1]})
    y = pd.Series([0, 1, 2, 3])
    model = LinearRegression().fit(X, y)

    imp_df = compute_permutation_importance(model, X, y, n_repeats=2, random_state=0)
    assert set(['feature', 'importance_mean', 'importance_std', 'importance_mean_minus_std']) <= set(getattr(imp_df, 'columns', []))
    assert len(imp_df) == 2

