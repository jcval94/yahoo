import pandas as pd

from src import preprocess


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
    monkeypatch.setattr(preprocess, 'generate_sample_data', lambda start, periods=30: sample_df)
    data = preprocess.extract_data(['BBB'], '2020-01-01')
    assert data['BBB'].equals(sample_df)


def test_extract_data_fallback_requests_longer_sample(monkeypatch):
    def raise_download(*args, **kwargs):
        raise RuntimeError('fail')

    captured = {}

    def fake_sample(start, periods=30):
        captured['periods'] = periods
        return pd.DataFrame({'Close': [1] * periods, 'Volume': [1] * periods})

    monkeypatch.setattr(preprocess.yf, 'download', raise_download)
    monkeypatch.setattr(preprocess, 'generate_sample_data', fake_sample)

    preprocess.extract_data(['BBB'], '2015-01-01')

    assert captured['periods'] >= 260


def test_enrich_features(monkeypatch):
    idx = pd.date_range('2024-01-01', periods=2, freq='D')
    df = pd.DataFrame({'Close': [1, 2], 'Open': [1, 2], 'High': [1, 2], 'Low': [1, 2], 'Volume': [1, 1]}, index=idx)

    def fake_add_technical(d):
        return d.assign(foo=1)

    def fake_load_exogenous(config, index, start, end):
        return pd.DataFrame({'vix_close': [10.0, 11.0]}, index=index)

    monkeypatch.setattr('src.features.add_technical_indicators', fake_add_technical)
    monkeypatch.setattr(preprocess, 'load_exogenous_factors', fake_load_exogenous)

    result = preprocess.enrich_features(df, config={'exogenous': {'enabled': True, 'use_vix': True}})
    assert 'foo' in result.columns
    assert 'vix_close' in result.columns


def test_preprocess_data(monkeypatch):
    data = {'AAA': pd.DataFrame({'Close': [1], 'Volume': [1]})}

    def fake_enrich(df, config=None):
        return df.assign(bar=2)

    monkeypatch.setattr(preprocess, 'enrich_features', fake_enrich)
    result = preprocess.preprocess_data(data, config={})
    assert 'AAA' in result
    assert 'bar' in result['AAA'].columns


def test_merge_exogenous_features_left_join_with_controlled_ffill():
    index = pd.to_datetime([
        '2024-01-01 09:30:00',
        '2024-01-01 10:00:00',
        '2024-01-01 10:30:00',
        '2024-01-01 11:00:00',
    ])
    base = pd.DataFrame({'Close': [100, 101, 102, 103]}, index=index)

    exog = pd.DataFrame(
        {'vix_close': [20.0]},
        index=pd.to_datetime(['2024-01-01 09:30:00']),
    )

    result = preprocess.merge_exogenous_features(base, exog, max_ffill_steps=2)

    assert result['vix_close'].iloc[0] == 20.0
    assert result['vix_close'].iloc[1] == 20.0
    assert result['vix_close'].iloc[2] == 20.0
    assert pd.isna(result['vix_close'].iloc[3])


def test_merge_exogenous_features_keeps_missing_when_no_previous_value():
    index = pd.date_range('2024-01-01', periods=3, freq='D')
    base = pd.DataFrame({'Close': [1, 2, 3]}, index=index)
    exog = pd.DataFrame({'is_fomc_day': [1]}, index=[pd.Timestamp('2024-01-02')])

    result = preprocess.merge_exogenous_features(base, exog, max_ffill_steps=1)

    assert pd.isna(result.loc['2024-01-01', 'is_fomc_day'])
    assert result.loc['2024-01-02', 'is_fomc_day'] == 1
    assert result.loc['2024-01-03', 'is_fomc_day'] == 1
