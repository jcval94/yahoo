"""Model evaluation and drift detection."""
import logging
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
    r2_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    """Return an expanded set of regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "EVS": evs,
    }
    logger.info("Evaluation metrics: %s", metrics)
    return {k: round(v, 4) for k, v in metrics.items()}


def detect_drift(prev: Sequence[float], curr: Sequence[float], threshold: float = 0.1) -> bool:
    """Simple drift detection comparing prediction distributions."""
    prev = np.array(prev)
    curr = np.array(curr)
    diff = np.abs(prev - curr).mean()
    return diff > threshold
