"""LSTM training and inference utilities.

When TensorFlow is available, train a small Keras LSTM.
When it is not available, fall back to a lightweight MLP surrogate so the
`lstm` pipeline remains operational and produces predictions.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import math
import time
from typing import Mapping, Sequence

import numpy as np
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

logger = logging.getLogger(__name__)

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

if TF_AVAILABLE:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras import callbacks, layers, regularizers

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    tf = None
    K = None
    callbacks = None
    layers = None
    regularizers = None


class LSTMSurrogateRegressor:
    """Sklearn fallback used when TensorFlow is unavailable."""

    def __init__(self, *, hidden_units: int = 16, l2_reg: float = 0.001) -> None:
        alpha = max(float(l2_reg), 1e-6)
        self.model = MLPRegressor(
            hidden_layer_sizes=(int(hidden_units),),
            activation="tanh",
            solver="adam",
            alpha=alpha,
            batch_size=64,
            max_iter=300,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMSurrogateRegressor":
        self.model.fit(_prepare_2d(X), np.asarray(y, dtype=np.float32))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(_prepare_2d(X))


def _prepare_2d(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _make_ds(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool = False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=42)
    return ds.cache().batch(batch, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def _build_lstm(input_shape: tuple[int, int], units: int, l2_reg: float = 0.0):
    reg = regularizers.l2(l2_reg) if l2_reg else None
    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(
                units,
                activation="tanh",
                recurrent_activation="sigmoid",
                kernel_regularizer=reg,
                recurrent_regularizer=reg,
            ),
            layers.Dense(1, kernel_regularizer=reg, dtype="float32"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mae", metrics=["mae"])
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_space: Mapping[str, Sequence] | None = None,
    n_iter: int = 2,
    cv_splits: int = 3,
    verbose: int = 0,
):
    """Train LSTM (TensorFlow) or surrogate model (sklearn fallback)."""
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)

    if param_space is None:
        param_space = {
            "units": [16, 32],
            "batch": [64],
            "epochs": [10],
            "l2_reg": [0.0, 1e-3],
        }

    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available. Using LSTMSurrogateRegressor fallback.")
        best = next(iter(ParameterSampler(param_space, n_iter=1, random_state=42)))
        surrogate = LSTMSurrogateRegressor(hidden_units=int(best.get("units", 16)), l2_reg=float(best.get("l2_reg", 1e-3)))
        surrogate.fit(X, y)
        return surrogate

    t0 = time.perf_counter()
    logger.info("⚙️  LSTM training (tensorflow backend)")

    if X.ndim == 2:
        X = X[:, None, :]

    best_params, best_score = None, math.inf
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    for params in ParameterSampler(param_space, n_iter=n_iter, random_state=42):
        fold_scores = []
        for tr_idx, val_idx in tscv.split(X):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = _build_lstm(X_tr.shape[1:], params["units"], params["l2_reg"])
            model.fit(
                _make_ds(X_tr, y_tr, params["batch"], shuffle=True),
                validation_data=_make_ds(X_val, y_val, params["batch"]),
                epochs=params["epochs"],
                verbose=verbose,
                callbacks=[callbacks.EarlyStopping(patience=4, monitor="val_mae", baseline=best_score, restore_best_weights=True)],
            )

            fold_mae = model.evaluate(_make_ds(X_val, y_val, params["batch"]), verbose=0)[0]
            fold_scores.append(fold_mae)
            K.clear_session()
            gc.collect()

        avg = float(np.mean(fold_scores))
        if avg < best_score:
            best_score, best_params = avg, params

    if best_params is None:
        raise RuntimeError("Random search sin resultados válidos.")

    final_model = _build_lstm(X.shape[1:], best_params["units"], best_params["l2_reg"])
    final_model.fit(
        _make_ds(X, y, best_params["batch"], shuffle=True),
        epochs=best_params["epochs"],
        verbose=verbose,
        callbacks=[callbacks.EarlyStopping(patience=6, monitor="loss", restore_best_weights=True)],
    )
    logger.info("✅ LSTM training done in %.1f s", time.perf_counter() - t0)
    return final_model


def prepare_prediction_data(X_new: np.ndarray) -> np.ndarray:
    X = np.asarray(X_new, dtype=np.float32)
    if X.ndim == 2:
        X = X[:, None, :]
    return X


def predict_lstm(model, X_new: np.ndarray) -> np.ndarray:
    if TF_AVAILABLE and tf is not None and isinstance(model, tf.keras.Model):
        X = prepare_prediction_data(X_new)
        return np.asarray(model.predict(X, verbose=0)).ravel()
    return np.asarray(model.predict(_prepare_2d(X_new))).ravel()
