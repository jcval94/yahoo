"""
Utility functions to train and use an LSTM model with proper
time-series CV and resource management.
"""

from __future__ import annotations
import logging, time, gc
from typing import Any, Sequence, Mapping

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, backend as K
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

logger = logging.getLogger(__name__)


def _build_lstm(
    input_shape: tuple[int, int],
    units: int = 32,
    dropout: float = 0.0,
    l2_reg: float = 0.0,
) -> tf.keras.Model:
    """Factory que devuelve un modelo LSTM compilado."""
    reg = regularizers.l2(l2_reg) if l2_reg else None

    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(units, dropout=dropout,
                        kernel_regularizer=reg,
                        recurrent_regularizer=reg,
                        activation="relu"),
            layers.Dense(1, kernel_regularizer=reg),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Mapping[str, Sequence] | None = None,
    cv: int | TimeSeriesSplit = 5,
    random_state: int | None = 42,
    verbose: int = 0,
) -> tf.keras.Model:
    t0 = time.perf_counter()
    logger.info("Training LSTM model with CV")

    if param_grid is None:  # grid por defecto
        param_grid = {
            "units": [32, 64],
            "epochs": [10],
            "dropout": [0.0, 0.2],
            "l2_reg": [0.0, 1e-3],
            "batch_size": [32],
        }

    # ---------- Pre-procesamiento seguro ----------
    try:
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
    except Exception as exc:
        logger.exception("Falló la conversión de X/y a numpy: %s", exc)
        raise

    if X.ndim == 2:        # añade dimensión temporal
        X = X[:, None, :]

    best_params, best_score = None, np.inf
    splitter = TimeSeriesSplit(n_splits=cv) if isinstance(cv, int) else cv

    # ---------- Búsqueda en rejilla con try/except ----------
    try:
        for params in ParameterGrid(param_grid):
            fold_scores = []
            for tr_idx, val_idx in splitter.split(X):
                try:
                    X_tr, X_val = X[tr_idx], X[val_idx]
                    y_tr, y_val = y[tr_idx], y[val_idx]

                    model = _build_lstm(
                        input_shape=X_tr.shape[1:],
                        units=params["units"],
                        dropout=params["dropout"],
                        l2_reg=params["l2_reg"],
                    )

                    es = callbacks.EarlyStopping(
                        patience=3, monitor="val_loss", restore_best_weights=True
                    )

                    model.fit(
                        X_tr,
                        y_tr,
                        validation_data=(X_val, y_val),
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                        verbose=verbose,
                    )

                    val_pred = model.predict(X_val, verbose=0).ravel()
                    mae = np.mean(np.abs(y_val - val_pred))
                    fold_scores.append(mae)

                except Exception as fold_exc:
                    logger.exception(
                        "Error en fold con params %s: %s. Se omite el fold.",
                        params,
                        fold_exc,
                    )
                    # opcional: fold_scores.append(np.inf)

                finally:
                    tf.keras.backend.clear_session()
                    gc.collect()

            avg_mae = float(np.mean(fold_scores))
            logger.debug("Params %s → MAE %.4f", params, avg_mae)

            if avg_mae < best_score:
                best_score, best_params = avg_mae, params

    except Exception as grid_exc:
        logger.exception("Error durante la búsqueda de hiperparámetros: %s", grid_exc)
        raise

    if best_params is None:
        raise RuntimeError("No se encontró ningún conjunto de parámetros válido.")

    logger.info("Best params: %s (MAE=%.4f)", best_params, best_score)

    # ---------- Entrenamiento final ----------
    try:
        final_model = _build_lstm(
            input_shape=X.shape[1:],
            units=best_params["units"],
            dropout=best_params["dropout"],
            l2_reg=best_params["l2_reg"],
        )
        es_final = callbacks.EarlyStopping(
            patience=3, monitor="loss", restore_best_weights=True
        )

        final_model.fit(
            X,
            y,
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            verbose=verbose,
            callbacks=[es_final],
        )

    except Exception as final_exc:
        logger.exception("Error al entrenar el modelo final: %s", final_exc)
        raise

    logger.info("Finished in %.1f s", time.perf_counter() - t0)
    return final_model


def predict_lstm(model: tf.keras.Model, X_new: np.ndarray) -> np.ndarray:
    """Predicciones con manejo de errores."""
    try:
        X = np.asarray(X_new, dtype=float)
        if X.ndim == 2:
            X = X[:, None, :]
        preds = model.predict(X, verbose=0).ravel()
        return preds
    except Exception as exc:
        logger.exception("Error en predict_lstm: %s", exc)
        raise
        
