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


def test_edge_evaluation_no_file(tmp_path):
    df = pd.DataFrame({'Close': [1, 2]}, index=pd.date_range('2020-01-01', periods=2))

    PRED_PATH = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'predict.py'
    spec = importlib.util.spec_from_file_location('predict', PRED_PATH)
    predict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict)

    predict.RESULTS_DIR = tmp_path

    result = predict.evaluate_edge_predictions({'TEST': df}, tmp_path / 'missing.csv')

    assert result is None
    metrics_dir = tmp_path / 'metrics'
    assert not any(metrics_dir.glob('edge_daily_*'))
