"""Utility functions to train and use an LSTM model."""
import logging
import time
from typing import Any

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


def train_lstm(X_train, y_train, **kwargs) -> Any:
    """Train a simple LSTM model."""
    start = time.perf_counter()
    logger.info("Training LSTM model")
    try:
        X = np.asarray(X_train).astype(float)
        y = np.asarray(y_train).astype(float)
        X = np.expand_dims(X, axis=1)
        model = keras.Sequential([
            layers.Input(shape=(X.shape[1], X.shape[2])),
            layers.LSTM(32, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=kwargs.get("epochs", 5), verbose=0)
    except Exception:
        logger.exception("Error while training LSTM")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("LSTM training finished in %.2f seconds", duration)
    return model


def predict_lstm(model: Any, X) -> Any:
    """Make predictions with a trained LSTM model."""
    start = time.perf_counter()
    logger.info("Running LSTM prediction")
    try:
        X = np.asarray(X).astype(float)
        X = np.expand_dims(X, axis=1)
        preds = model.predict(X, verbose=0).flatten()
    except Exception:
        logger.exception("Error during LSTM prediction")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Prediction finished in %.2f seconds", duration)
    return preds
