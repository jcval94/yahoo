import importlib.util
import pathlib
import pytest

pd = pytest.importorskip("pandas")

PRED_PATH = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'predict.py'
spec = importlib.util.spec_from_file_location('predict', PRED_PATH)
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)


def test_predictions_file_and_order(tmp_path):
    df = pd.DataFrame({'Close': [1, 2, 3]}, index=pd.date_range('2020-01-01', periods=3))
    models = {'TEST_dummy': predict._NaiveModel()}
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
    models = {'TEST_dummy': predict._NaiveModel()}
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
    mdf = pd.read_csv(metrics_file_edge)
    assert 'pred' in mdf.columns
    assert 'real' in mdf.columns
    assert 'pred_delta' in mdf.columns
    assert 'real_delta' in mdf.columns
    assert 'pred_delta_pct' in mdf.columns
    assert 'real_delta_pct' in mdf.columns
    assert 'direction' in mdf.columns
    # verify actual value corresponds to prediction date
    assert (mdf['Predicted'] == '2020-01-06').all()
    assert (mdf['real'] == 6).all()


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
