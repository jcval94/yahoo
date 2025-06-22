"""ARIMA model utilities."""
import logging
import time
from typing import Any, Sequence

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


class ARIMAModel:
    """Wrapper to provide a predict method similar to scikit-learn."""

    def __init__(self, results):
        self.results = results

    def predict(self, X=None):
        preds = self.results.predict()
        if X is not None:
            try:
                n = len(X)
                if len(preds) > n:
                    preds = preds[-n:]
            except Exception:
                pass
        return preds


def train_arima(y_train: Sequence, order=(1, 0, 0), **kwargs) -> Any:
    """Train an ARIMA model on a univariate series."""
    start = time.perf_counter()
    logger.info("Training ARIMA model")

    try:
        model = ARIMA(np.asarray(y_train).astype(float), order=order, **kwargs)
        results = model.fit()
    except Exception:
        logger.exception("Error while training ARIMA")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("ARIMA training finished in %.2f seconds", duration)

    return ARIMAModel(results)
