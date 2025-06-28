import importlib.util
import pathlib

OPT_PATH = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'portfolio' / 'optimize.py'
spec = importlib.util.spec_from_file_location('optimize', OPT_PATH)
optimize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimize)


def test_optimize_basic():
    data = {
        'A': [0.1, 0.05, -0.02],
        'B': [0.02, 0.03, 0.04],
    }
    weights = optimize.optimize_portfolio(data)
    assert isinstance(weights, dict)
    assert set(weights) == {'A', 'B'}
    total = sum(weights.values())
    assert abs(total - 1) < 1e-6
    assert weights['B'] > weights['A']
    assert round(weights['A'], 3) == 0.142
    assert round(weights['B'], 3) == 0.858
