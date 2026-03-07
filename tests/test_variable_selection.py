import importlib
import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

if not hasattr(pd.DataFrame, 'corr'):
    pytest.skip("pandas not installed", allow_module_level=True)

vs = importlib.import_module('src.variable_selection')
select_features = vs.select_features
select_features_rf_cv = getattr(vs, 'select_features_rf_cv')


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


def test_select_features_rf_cv_basic():
    n = 60
    rng = np.random.default_rng(0)
    x1 = np.linspace(0, 1, n)
    x2 = x1 * 0.95 + 0.05
    x3 = rng.normal(size=n)
    y = 0.8 * x1 + 0.2 * x3
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'target': y})
    selected = select_features_rf_cv(df, 'target', cv=2, corr_threshold=0.8)
    assert sum(f in selected for f in ['x1', 'x2']) == 1
    assert 'x3' in selected
    assert len(selected) <= int(np.sqrt(3)) + 7


def test_select_features_keeps_event_predictors():
    n = 40
    x = np.linspace(0, 1, n)
    target = np.sin(x)
    df = pd.DataFrame(
        {
            'x': x,
            'gap_pct': np.zeros(n),
            'overnight_return': np.zeros(n),
            'open_to_close_return': np.zeros(n),
            'drawdown_from_prev_close': np.zeros(n),
            'recovery_bars_5pct': np.zeros(n),
            'recovery_bars_10pct': np.zeros(n),
            'recovery_bars_20pct': np.zeros(n),
            'target': target,
        }
    )
    selected = select_features(df, 'target', relevance_threshold=0.99)
    for col in [
        'gap_pct',
        'overnight_return',
        'open_to_close_return',
        'drawdown_from_prev_close',
        'recovery_bars_5pct',
        'recovery_bars_10pct',
        'recovery_bars_20pct',
    ]:
        assert col in selected
