"""Random Forest utilities with temporal cross-validation."""
import logging
import time
from typing import Any

from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def train_rf(X_train, y_train, **kwargs) -> Any:
    start = time.perf_counter()
    logger.info("Training Random Forest model")
    try:
        model = RandomForestRegressor(**kwargs)
        model.fit(X_train, y_train)
    except Exception:
        logger.exception("Error while training Random Forest")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Random Forest training finished in %.2f seconds", duration)
    return model
