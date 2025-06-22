"""LightGBM utilities with simple cross-validation support."""
import logging
import time
from typing import Any, Dict, Sequence

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)


def train_lgbm(
    X_train,
    y_train,
    param_grid: Dict[str, Sequence] | None = None,
    cv: int = 3,
    **kwargs,
) -> Any:
    """Train a LightGBM model with optional cross-validation."""
    start = time.perf_counter()
    logger.info("Training LightGBM model")

    if param_grid is None:
        param_grid = {
            "n_estimators": [50],
            "max_depth": [3],
        }

    try:
        base_model = LGBMRegressor(random_state=42, **kwargs)
        splitter = TimeSeriesSplit(n_splits=cv)
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=splitter,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        logger.info("LGBM best params: %s", search.best_params_)
    except Exception:
        logger.exception("Error while training LightGBM")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("LightGBM training finished in %.2f seconds", duration)

    return model
