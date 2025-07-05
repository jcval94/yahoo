"""Simple ridge regression utilities."""
import logging
import time
from typing import Any, Union, Sequence

from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    BaseCrossValidator,
)

logger = logging.getLogger(__name__)


def train_linear(
    X_train,
    y_train,
    cv: Union[int, BaseCrossValidator] = 5,
    alphas: Sequence[float] | None = None,
    n_iter: int = 10,
    **kwargs,
) -> Any:
    """Train a ridge regression model with basic cross-validation.

    Parameters
    ----------
    alphas
        Optional list of ``alpha`` values to evaluate during the random search.
    n_iter
        Maximum number of parameter settings sampled by
        :class:`RandomizedSearchCV`.
    """
    start = time.perf_counter()
    logger.info("Training Ridge Regression model")

    try:
        if alphas is None:
            alphas = [0.1, 1.0, 10.0]

        splitter = TimeSeriesSplit(n_splits=cv) if isinstance(cv, int) else cv
        base_model = Ridge(solver="svd", **kwargs)
        search = RandomizedSearchCV(
            base_model,
            param_distributions={"alpha": alphas},
            cv=splitter,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            n_iter=n_iter,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        logger.info("Ridge best params: %s", search.best_params_)
    except Exception:
        logger.exception("Error while training Ridge Regression")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Ridge Regression training finished in %.2f seconds", duration)

    return model
