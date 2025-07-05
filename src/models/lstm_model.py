"""
Fast LSTM training utils  🚀
✓ CuDNN-compatible
✓ mixed-precision
✓ tf.data pipeline
✓ random search + walk-forward CV
"""

from __future__ import annotations
import logging, time, gc, math, random
from typing import Any, Mapping, Sequence, Iterable

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, backend as K
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler

logger = logging.getLogger(__name__)

# ---------- Configuración global ----------
tf.keras.mixed_precision.set_global_policy("mixed_float16")
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# ---------- Dataset helper ----------
def _make_ds(
    X: np.ndarray,
    y: np.ndarray,
    batch: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Return a ``tf.data`` pipeline with a fixed batch shape."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=42)
    return (
        ds.cache()
        .batch(batch, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

# ---------- Model factory ----------
def _build_lstm(input_shape: tuple[int, int],
                units: int,
                l2_reg: float = 0.0) -> tf.keras.Model:
    reg = regularizers.l2(l2_reg) if l2_reg else None
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units,
                    activation="tanh",           # CuDNN
                    recurrent_activation="sigmoid",
                    kernel_regularizer=reg,
                    recurrent_regularizer=reg),
        layers.Dense(1, kernel_regularizer=reg, dtype="float32"),  # evita overflow
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="mae", metrics=["mae"])
    return model

# ---------- Entrenamiento ----------
def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_space: Mapping[str, Sequence] | None = None,
    n_iter: int = 4,
    cv_splits: int = 3,
    verbose: int = 0,
) -> tf.keras.Model:
    t0 = time.perf_counter()
    logger.info("⚙️  LSTM training (fast)")

    # -- defaults -------------------------
    if param_space is None:
        param_space = {
            "units": [32, 64, 96],
            "batch": [64],
            "epochs": [30],
            "l2_reg": [0.0, 1e-3],
        }

    # -- reshape --------------------------
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    if X.ndim == 2:
        X = X[:, None, :]                         # (samples, 1, features)

    best_params, best_score = None, math.inf
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # -- random search --------------------
    for params in ParameterSampler(param_space, n_iter=n_iter, random_state=42):
        fold_scores = []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = _build_lstm(X_tr.shape[1:], params["units"], params["l2_reg"])
            es = callbacks.EarlyStopping(
                patience=4, monitor="val_mae",
                baseline=best_score,               # abandona antes si es peor
                restore_best_weights=True
            )

            model.fit(
                _make_ds(X_tr, y_tr, params["batch"], shuffle=True),
                validation_data=_make_ds(X_val, y_val, params["batch"]),
                epochs=params["epochs"],
                verbose=verbose,
            )

            fold_mae = model.evaluate(
                _make_ds(X_val, y_val, params["batch"]),
                verbose=0,
            )[0]
            fold_scores.append(fold_mae)

            # limpieza
            K.clear_session(); gc.collect()

        avg = float(np.mean(fold_scores))
        logger.debug("Params %s → MAE %.3f", params, avg)
        if avg < best_score:
            best_score, best_params = avg, params

    if best_params is None:
        raise RuntimeError("Random search sin resultados válidos.")

    logger.info("🥇 Best %s  |  MAE=%.3f", best_params, best_score)

    # -- entrenamiento final --------------
    final_model = _build_lstm(X.shape[1:], best_params["units"], best_params["l2_reg"])
    es_final = callbacks.EarlyStopping(
        patience=6, monitor="loss", restore_best_weights=True
    )
    final_model.fit(
        _make_ds(X, y, best_params["batch"], shuffle=True),
        epochs=best_params["epochs"],
        verbose=verbose,
        callbacks=[es_final],
    )

    logger.info("✅ Done in %.1f s", time.perf_counter() - t0)
    return final_model


def prepare_prediction_data(X_new: np.ndarray) -> np.ndarray:
    """Format new samples for inference with the trained LSTM."""
    X = np.asarray(X_new, dtype=np.float32)
    if X.ndim == 2:
        X = X[:, None, :]
    return X


def predict_lstm(model: tf.keras.Model, X_new: np.ndarray) -> np.ndarray:
    X = prepare_prediction_data(X_new)
    return model.predict(X, verbose=0).ravel()
