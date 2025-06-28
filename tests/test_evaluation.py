import importlib.util
import pathlib
import sys
import types
import math

# Minimal numpy stub for module import if numpy is missing
if 'numpy' not in sys.modules:
    np_stub = types.ModuleType('numpy')
    np_stub.array = lambda x: list(x)
    np_stub.abs = lambda x: [abs(i) for i in x]
    np_stub.mean = lambda x: sum(x)/len(x)
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

# Provide stub sklearn.metrics if scikit-learn is unavailable
try:
    import sklearn.metrics as metrics
except ModuleNotFoundError:
    metrics = types.ModuleType('sklearn.metrics')
    def mae(y_true, y_pred):
        return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
    def mse(y_true, y_pred):
        return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
    def rmse(y_true, y_pred):
        return math.sqrt(mse(y_true, y_pred))
    def mape(y_true, y_pred):
        return sum(abs((t - p) / t) for t, p in zip(y_true, y_pred)) / len(y_true)
    def r2(y_true, y_pred):
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
        ss_tot = sum((t - mean_y) ** 2 for t in y_true)
        return 1 - ss_res / ss_tot
    def evs(y_true, y_pred):
        diff = [t - p for t, p in zip(y_true, y_pred)]
        mean_diff = sum(diff) / len(diff)
        var_diff = sum((d - mean_diff) ** 2 for d in diff) / len(diff)
        mean_true = sum(y_true) / len(y_true)
        var_true = sum((t - mean_true) ** 2 for t in y_true) / len(y_true)
        return 1 - var_diff / var_true
    metrics.mean_absolute_error = mae
    metrics.mean_squared_error = mse
    metrics.root_mean_squared_error = rmse
    metrics.mean_absolute_percentage_error = mape
    metrics.r2_score = r2
    metrics.explained_variance_score = evs
    sklearn_mod = types.ModuleType('sklearn')
    sklearn_mod.metrics = metrics
    sys.modules.setdefault('sklearn', sklearn_mod)
    sys.modules['sklearn.metrics'] = metrics

# Load evaluation module directly
EVAL_PATH = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'evaluation.py'
spec = importlib.util.spec_from_file_location('evaluation', EVAL_PATH)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

def test_evaluate_predictions_basic():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 4]
    result = evaluation.evaluate_predictions(y_true, y_pred)
    expected = {
        'MAE': round(metrics.mean_absolute_error(y_true, y_pred), 4),
        'MSE': round(metrics.mean_squared_error(y_true, y_pred), 4),
        'RMSE': round(metrics.root_mean_squared_error(y_true, y_pred), 4),
        'MAPE': round(metrics.mean_absolute_percentage_error(y_true, y_pred), 4),
        'R2': round(metrics.r2_score(y_true, y_pred), 4),
        'EVS': round(metrics.explained_variance_score(y_true, y_pred), 4),
    }
    assert result == expected
