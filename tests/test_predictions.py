import pytest

pd = pytest.importorskip("pandas")

from src import predict


def test_predictions_file_and_order(tmp_path):
    df = pd.DataFrame({'Close': [1, 2, 3]}, index=pd.date_range('2020-01-01', periods=3))
    models = {'TEST_daily_dummy': predict._NaiveModel()}
    predict.RESULTS_DIR = tmp_path
    predict.RUN_TIMESTAMP = '2025-07-07T00:00:00+00:00'
    result = predict.run_predictions(models, {'TEST': df}, frequency='daily')
    pred_file = tmp_path / 'predicts' / '2025-07-07_daily_predictions.csv'
    assert pred_file.exists()
    assert result.columns[-1] == 'parameters'


def test_edge_prediction_and_metrics(tmp_path):
    df = pd.DataFrame(
        {'Close': [1, 2, 3, 4, 5, 6]},
        index=pd.date_range('2020-01-01', periods=6)
    )
    models = {'TEST_daily_dummy': predict._NaiveModel()}
    predict.RESULTS_DIR = tmp_path
    predict.RUN_TIMESTAMP = '2025-07-08T00:00:00+00:00'

    result = predict.run_predictions(models, {'TEST': df[:-1]}, frequency='daily')
    edge_file = predict.save_edge_predictions(result)
    assert edge_file.exists()

    predict.evaluate_edge_predictions({'TEST': df}, edge_file)
    metrics_file_edge = tmp_path / 'edge_metrics' / 'edge_metrics_2020-01-06.csv'
    assert metrics_file_edge.exists()
    metrics_file_main = tmp_path / 'metrics' / 'edge_metrics_2020-01-06.csv'
    assert metrics_file_main.exists()
    metrics_file_by_session = tmp_path / 'metrics' / 'edge_metrics_2020-01-06_by_session.csv'
    metrics_file_by_event = tmp_path / 'metrics' / 'edge_metrics_2020-01-06_by_event.csv'
    assert metrics_file_by_session.exists()
    assert metrics_file_by_event.exists()
    mdf = pd.read_csv(metrics_file_edge)
    assert 'pred' in mdf.columns
    assert 'real' in mdf.columns
    assert 'pred_delta' in mdf.columns
    assert 'real_delta' in mdf.columns
    assert 'pred_inc' in mdf.columns
    assert 'real_inc' in mdf.columns
    assert 'direction' in mdf.columns
    # verify actual value corresponds to prediction date
    assert (mdf['Predicted'] == '2020-01-06').all()
    assert (mdf['real'] == 6).all()

    session_df = pd.read_csv(metrics_file_by_session)
    event_df = pd.read_csv(metrics_file_by_event)
    assert not session_df.empty
    assert not event_df.empty



def test_save_edge_predictions_keeps_all_models(tmp_path):
    predict.RESULTS_DIR = tmp_path
    predict.RUN_TIMESTAMP = '2025-07-10T00:00:00+00:00'

    result = pd.DataFrame(
        [
            {'ticker': 'AAA', 'model': 'arima', 'pred': 10.0, 'Predicted': '2025-07-11'},
            {'ticker': 'AAA', 'model': 'rf', 'pred': 11.0, 'Predicted': '2025-07-11'},
            {'ticker': 'AAA', 'model': 'xgb', 'pred': 12.0, 'Predicted': '2025-07-11'},
            {'ticker': 'AAA', 'model': 'lgbm', 'pred': 13.0, 'Predicted': '2025-07-11'},
        ]
    )

    metrics_dir = tmp_path / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        [
            {'model': 'AAA_rf', 'dataset': 'test', 'RMSE': 1.0},
            {'model': 'AAA_xgb', 'dataset': 'test', 'RMSE': 1.1},
            {'model': 'AAA_rf', 'dataset': 'test', 'RMSE': 1.2},
            {'model': 'AAA_arima', 'dataset': 'test', 'RMSE': 1.3},
        ]
    )
    metrics.to_csv(metrics_dir / 'metrics_daily_2025-07-10.csv', index=False)

    edge_file = predict.save_edge_predictions(result)
    edge_df = pd.read_csv(edge_file)

    per_model = edge_df[(edge_df['ticker'] == 'AAA') & (edge_df['model'] != 'Top3Ensamble')]
    assert set(per_model['model']) == {'arima', 'rf', 'xgb', 'lgbm'}

    ens = edge_df[(edge_df['ticker'] == 'AAA') & (edge_df['model'] == 'Top3Ensamble')]
    assert len(ens) == 1
    assert ens['pred'].iloc[0] == pytest.approx((10.0 + 11.0 + 12.0) / 3)

def test_edge_evaluation_no_file(tmp_path):
    df = pd.DataFrame({'Close': [1, 2, 3]}, index=pd.date_range('2020-01-01', periods=3))
    predict.RESULTS_DIR = tmp_path
    predict.RUN_TIMESTAMP = '2025-07-09T00:00:00+00:00'

    result = predict.evaluate_edge_predictions({'TEST': df}, tmp_path / 'missing.csv')
    assert result is None
    metrics_files = list((tmp_path / 'edge_metrics').glob('edge_metrics_*'))
    assert not metrics_files
    metrics_main_files = list((tmp_path / 'metrics').glob('edge_metrics_*'))
    assert not metrics_main_files


def test_load_models_filters_stale_tickers(tmp_path, monkeypatch):
    class _DummyModel:
        def predict(self, X):
            return [0]

    keep = tmp_path / "NVDA_daily_rf_aaaaaaaaaa.joblib"
    drop = tmp_path / "BIMBO_daily_rf_bbbbbbbbbb.joblib"
    keep.write_text("ok")
    drop.write_text("ok")

    monkeypatch.setattr(predict, "CONFIG", {"etfs": ["NVDA"]})
    monkeypatch.setattr(predict, "load_with_schema", lambda path: (_DummyModel(), ["Close"], "hash"))

    models = predict.load_models(tmp_path)

    assert set(models) == {"NVDA_daily_rf_aaaaaaaaaa"}
