import importlib.util
import pathlib
import sys
import types

# Minimal numpy and pandas stubs so utils can be imported without dependencies
if 'numpy' not in sys.modules:
    np_stub = types.ModuleType('numpy')
    np_stub.linspace = lambda start, stop, num: [start + (stop - start)/(num-1)*i for i in range(num)] if num > 1 else [start]
    class _Random:
        def randint(self, low, high=None, size=None):
            import random
            if high is None:
                high = low
                low = 0
            if size is None:
                return random.randint(low, high - 1)
            return [random.randint(low, high - 1) for _ in range(size)]
    np_stub.random = _Random()
    sys.modules['numpy'] = np_stub

if 'pandas' not in sys.modules:
    pd_stub = types.ModuleType('pandas')
    def date_range(start, periods, freq=None):
        from datetime import datetime, timedelta
        start_dt = datetime.fromisoformat(start)
        return [start_dt + timedelta(days=i) for i in range(periods)]
    class DataFrame(dict):
        def __init__(self, data, index=None):
            super().__init__(data)
            self.index = index
        def head(self, n):
            return self
        @property
        def empty(self):
            return False
        @property
        def shape(self):
            return (len(self.index), len(self))
        def to_string(self):
            return 'df'
    pd_stub.date_range = date_range
    pd_stub.DataFrame = DataFrame
    sys.modules['pandas'] = pd_stub

# Provide stub sklearn.model_selection if scikit-learn is unavailable
try:
    from sklearn.model_selection import TimeSeriesSplit
except ModuleNotFoundError:
    ms = types.ModuleType('sklearn.model_selection')
    class TimeSeriesSplit:
        def __init__(self, n_splits=5, test_size=None, max_train_size=None):
            self.n_splits = n_splits
            self.test_size = test_size
            self.max_train_size = max_train_size
    ms.TimeSeriesSplit = TimeSeriesSplit
    sklearn_mod = types.ModuleType('sklearn')
    sklearn_mod.model_selection = ms
    sys.modules.setdefault('sklearn', sklearn_mod)
    sys.modules['sklearn.model_selection'] = ms

# Load utils module directly
UTILS_PATH = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'utils' / '__init__.py'
spec = importlib.util.spec_from_file_location('utils', UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
rolling_cv = utils.rolling_cv
hybrid_cv_split = utils.hybrid_cv_split


def test_rolling_cv_params():
    cv = rolling_cv(n_samples=100, train_size=60, horizon=1, max_splits=5)
    assert isinstance(cv, TimeSeriesSplit)
    assert cv.n_splits == 5
    assert cv.test_size == 1
    assert cv.max_train_size == 60


def test_rolling_cv_small_samples():
    cv = rolling_cv(n_samples=62, train_size=60, horizon=1, max_splits=5)
    assert cv.n_splits == 2


def test_rolling_cv_horizon_limit():
    cv = rolling_cv(n_samples=200, train_size=90, horizon=20, max_splits=10)
    expected = min(10, (200 - 90) // 20)
    assert cv.n_splits == expected
    assert cv.test_size == 20
    assert cv.max_train_size == 90

def test_hybrid_cv_split_basic():
    data = list(range(200))
    folds = list(hybrid_cv_split(data))
    assert 1 <= len(folds) <= 10

    train_idx, test_idx = folds[0]
    assert len(train_idx) == 90
    assert len(test_idx) == 1
    assert test_idx[0] - train_idx[-1] == 6

    if len(folds) > 1:
        first_test = folds[0][1][0]
        second_test = folds[1][1][0]
        assert second_test - first_test == 7
