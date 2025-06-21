"""XGBoost model utilities."""
import logging
import time
from typing import Any

from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def train_xgb(X_train, y_train, **kwargs) -> Any:
    start = time.perf_counter()
    logger.info("Training XGBoost model")
    try:
        model = XGBRegressor(**kwargs)
        model.fit(X_train, y_train)
    except Exception:
        logger.exception("Error while training XGBoost")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("XGBoost training finished in %.2f seconds", duration)
    return model
