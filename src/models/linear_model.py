"""Simple ridge regression utilities."""
import logging
import time
from typing import Any, Union, Sequence

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, BaseCrossValidator

logger = logging.getLogger(__name__)


def train_linear(
    X_train,
    y_train,
    cv: Union[int, BaseCrossValidator] = 5,
    alphas: Sequence[float] | None = None,
    **kwargs,
) -> Any:
    """Train a ridge regression model with basic cross-validation.

    Parameters
    ----------
    alphas
        Optional list of ``alpha`` values to evaluate. The value with the
        lowest validation error is used for the final model.
    """
    start = time.perf_counter()
    logger.info("Training Ridge Regression model")

    try:
        if alphas is None:
            alphas = [0.1, 1.0, 10.0]

        splitter = TimeSeriesSplit(n_splits=cv) if isinstance(cv, int) else cv
        best_alpha = alphas[0]
        best_score = float("inf")
        for alpha in alphas:
            model = Ridge(alpha=alpha, solver="svd", **kwargs)
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=splitter,
                scoring="neg_mean_absolute_error",
            )
            score = -scores.mean()
            if score < best_score:
                best_score = score
                best_alpha = alpha

        logger.info("Ridge best alpha: %s", best_alpha)

        model = Ridge(alpha=best_alpha, solver="svd", **kwargs)
        model.fit(X_train, y_train)
    except Exception:
        logger.exception("Error while training Ridge Regression")
        raise
    finally:
        duration = time.perf_counter() - start
        logger.info("Ridge Regression training finished in %.2f seconds", duration)

    return model
