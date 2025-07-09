import importlib.util
import pathlib
import pytest

pd = pytest.importorskip("pandas")

SEL_PATH = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'selection.py'
spec = importlib.util.spec_from_file_location('selection', SEL_PATH)
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def test_fetch_history_uses_yfinance(monkeypatch):
    called = {}

    def fake_download(ticker, start=None, end=None, progress=False, auto_adjust=False):
        called['args'] = (ticker, start, end, progress, auto_adjust)
        return pd.DataFrame({'Close': [1], 'Volume': [1]})

    monkeypatch.setattr(selection.yf, 'download', fake_download)
    df = selection._fetch_history('AAA', '2020-01-01', '2020-01-02')
    assert not df.empty
    assert called['args'] == ('AAA', '2020-01-01', '2020-01-02', False, False)


def test_select_tickers_basic(monkeypatch):
    r2_map = {1: 0.02, 2: 0.04, 3: 0.06, 4: -0.08, 5: -0.1}

    def fake_fetch(ticker, start, end):
        idx = int(ticker[1:])
        vol = 100 + idx
        r2 = r2_map.get(idx, 0.03)
        close = [100, 100, 100 * (1 + r2)]
        df = pd.DataFrame({'Close': close, 'Volume': [vol] * 3})
        return df

    monkeypatch.setattr(selection, '_fetch_history', fake_fetch)
    candidates = [f'T{i}' for i in range(1, 11)]
    selected = selection.select_tickers(candidates, '2023-01-01')
    expected = ['T10', 'T9', 'T8', 'T7', 'T6', 'T1', 'T2', 'T3', 'T4', 'T5']
    assert selected == expected
