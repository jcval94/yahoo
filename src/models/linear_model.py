"""Simple ridge regression utilities."""
import logging
import time
from typing import Any, Union

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, BaseCrossValidator

logger = logging.getLogger(__name__)


def train_linear(
    X_train,
    y_train,
    cv: Union[int, BaseCrossValidator] = 5,
    **kwargs,
) -> Any:
    """Train a ridge regression model with basic cross-validation."""
    start = time.perf_counter()
    logger.info("Training Ridge Regression model")

    try:
        model = Ridge(**kwargs)
        splitter = TimeSeriesSplit(n_splits=cv) if isinstance(cv, int) else cv
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=splitter,
            scoring="neg_mean_absolute_error",
        )
        logger.info("Ridge CV MAE: %.4f", -scores.mean())
        model.fit(X_train, y_train)
    except Exception:
        logger.exception("Error while training Ridge Regression")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Ridge Regression training finished in %.2f seconds", duration)

    return model
