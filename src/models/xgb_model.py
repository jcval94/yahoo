"""XGBoost model utilities with simple cross-validation support."""
import logging
import time
from typing import Any, Dict, Sequence, Union

from xgboost import XGBRegressor
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
    BaseCrossValidator,
)

logger = logging.getLogger(__name__)


def train_xgb(
    X_train,
    y_train,
    param_grid: Dict[str, Sequence] | None = None,
    cv: Union[int, BaseCrossValidator] = 5,
    **kwargs,
) -> Any:
    """Train an XGBoost model with optional cross-validation.

    Parameters
    ----------
    X_train, y_train
        Training features and target.
    param_grid
        Dictionary of parameter grids for :class:`GridSearchCV`.
        If ``None``, a small search space is used.
    cv
        Number of splits for :class:`TimeSeriesSplit`.
    kwargs
        Extra parameters passed directly to ``XGBRegressor``.
    """

    start = time.perf_counter()
    logger.info("Training XGBoost model")

    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }

    try:
        base_model = XGBRegressor(random_state=42, verbosity=0, **kwargs)
        splitter = TimeSeriesSplit(n_splits=cv) if isinstance(cv, int) else cv
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=splitter,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        logger.info("XGB best params: %s", search.best_params_)
    except Exception:
        logger.exception("Error while training XGBoost")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("XGBoost training finished in %.2f seconds", duration)

    return model
