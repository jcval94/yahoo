"""Utility functions to train and use an LSTM model with simple CV."""
import logging
import time
from typing import Any, Dict, Sequence, Union

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import TimeSeriesSplit, BaseCrossValidator

logger = logging.getLogger(__name__)


def train_lstm(
    X_train,
    y_train,
    param_grid: Dict[str, Sequence] | None = None,
    cv: Union[int, BaseCrossValidator] = 3,
    **kwargs,
) -> Any:
    """Train a simple LSTM using manual cross-validation."""

    start = time.perf_counter()
    logger.info("Training LSTM model")

    if param_grid is None:
        param_grid = {"units": [16], "epochs": [2]}

    X = np.asarray(X_train).astype(float)
    y = np.asarray(y_train).astype(float)
    X = np.expand_dims(X, axis=1)

    best_score = float("inf")
    best_params = {"units": 16, "epochs": 2}

    splitter = TimeSeriesSplit(n_splits=cv) if isinstance(cv, int) else cv

    try:
        for units in param_grid.get("units", [16]):
            for epochs in param_grid.get("epochs", [2]):
                scores = []
                for train_idx, val_idx in splitter.split(X):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    model = keras.Sequential([
                        layers.Input(shape=(X_tr.shape[1], X_tr.shape[2])),
                        layers.LSTM(units, activation="relu"),
                        layers.Dense(1),
                    ])
                    model.compile(optimizer="adam", loss="mse")
                    model.fit(X_tr, y_tr, epochs=epochs, verbose=0)
                    val_pred = model.predict(X_val, verbose=0).flatten()
                    scores.append(np.mean(np.abs(y_val - val_pred)))

                avg_score = float(np.mean(scores))
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {"units": units, "epochs": epochs}

        logger.info("LSTM best params: %s", best_params)

        # Train final model on full dataset
        model = keras.Sequential([
            layers.Input(shape=(X.shape[1], X.shape[2])),
            layers.LSTM(best_params["units"], activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=best_params["epochs"], verbose=0)
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
