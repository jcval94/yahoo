import importlib.util
import pathlib
import pytest

pd = pytest.importorskip("pandas")

PREP_PATH = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'preprocess.py'
spec = importlib.util.spec_from_file_location('preprocess', PREP_PATH)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)


def test_extract_data_success(monkeypatch):
    def fake_download(ticker, start=None, progress=False, auto_adjust=False):
        return pd.DataFrame({'Close': [1], 'Volume': [1]})

    monkeypatch.setattr(preprocess.yf, 'download', fake_download)
    data = preprocess.extract_data(['AAA'], '2020-01-01')
    assert 'AAA' in data
    assert isinstance(data['AAA'], pd.DataFrame)


def test_extract_data_fallback(monkeypatch):
    def raise_download(*args, **kwargs):
        raise RuntimeError('fail')

    sample_df = pd.DataFrame({'Close': [1], 'Volume': [1]})
    monkeypatch.setattr(preprocess.yf, 'download', raise_download)
    monkeypatch.setattr(preprocess, 'generate_sample_data', lambda start: sample_df)
    data = preprocess.extract_data(['BBB'], '2020-01-01')
    assert data['BBB'].equals(sample_df)


def test_enrich_features(monkeypatch):
    df = pd.DataFrame({'Close': [1, 2], 'Volume': [1, 1]})
    monkeypatch.setattr(preprocess, 'add_technical_indicators', lambda d: d.assign(foo=1))
    result = preprocess.enrich_features(df)
    assert 'foo' in result.columns


def test_preprocess_data(monkeypatch):
    data = {'AAA': pd.DataFrame({'Close': [1], 'Volume': [1]})}

    def fake_enrich(df):
        return df.assign(bar=2)

    monkeypatch.setattr(preprocess, 'enrich_features', fake_enrich)
    result = preprocess.preprocess_data(data)
    assert 'AAA' in result
    assert 'bar' in result['AAA'].columns
