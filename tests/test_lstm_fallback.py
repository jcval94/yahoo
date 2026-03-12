import numpy as np
import pytest

from src.models.lstm_model import TF_AVAILABLE, predict_lstm, train_lstm


def test_train_lstm_fallback_produces_predictions_without_tensorflow():
    if TF_AVAILABLE:
        pytest.skip("TensorFlow is installed; fallback path not exercised")

    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 6))
    y = X[:, 0] * 0.5 - X[:, 1] * 0.25

    model = train_lstm(X, y, n_iter=1, cv_splits=2)
    preds = predict_lstm(model, X[:5])

    assert preds.shape == (5,)
    assert np.isfinite(preds).all()
