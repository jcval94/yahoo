"""Random Forest utilities with simple cross-validation support."""
import logging
import time
from typing import Any, Dict, Sequence

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)


def train_rf(
    X_train,
    y_train,
    param_grid: Dict[str, Sequence] | None = None,
    cv: int = 3,
    **kwargs,
) -> Any:
    """Train a Random Forest with optional cross-validation.

    Parameters
    ----------
    X_train, y_train
        Training features and target.
    param_grid
        Dictionary of parameters for :class:`GridSearchCV`. If ``None``, a
        minimal grid with ``n_estimators`` and ``max_depth`` is used.
    cv
        Number of splits for :class:`TimeSeriesSplit`.
    kwargs
        Extra parameters passed directly to ``RandomForestRegressor``.
    """

    start = time.perf_counter()
    logger.info("Training Random Forest model")

    if param_grid is None:
        param_grid = {
            "n_estimators": [20],
            "max_depth": [3],
        }

    try:
        base_model = RandomForestRegressor(random_state=42, **kwargs)
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
        logger.info("RF best params: %s", search.best_params_)
    except Exception:
        logger.exception("Error while training Random Forest")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Random Forest training finished in %.2f seconds", duration)

    return model
