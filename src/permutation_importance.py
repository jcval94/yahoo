"""Utility to compute permutation feature importance."""
from __future__ import annotations

import pandas as pd

try:
    from sklearn.inspection import permutation_importance as _perm_importance
except Exception:  # pragma: no cover - scikit-learn may not be available
    _perm_importance = None

DEFAULT_EVENT_PREDICTORS = [
    "gap_pct",
    "overnight_return",
    "open_to_close_return",
    "drawdown_from_prev_close",
    "recovery_bars_5pct",
    "recovery_bars_10pct",
    "recovery_bars_20pct",
]


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str = "neg_mean_absolute_error",
):
    """Return permutation importance mean and std as a DataFrame.

    Parameters
    ----------
    model
        Fitted estimator with a ``predict`` method.
    X
        Feature matrix used for computing importances.
    y
        Target values corresponding to ``X``.
    n_repeats
        Number of random shuffles per feature.
    random_state
        Seed for the random generator.
    scoring
        Scoring strategy passed to ``permutation_importance``.
    """
    if _perm_importance is None:
        raise ImportError("scikit-learn is required for permutation importance")

    result = _perm_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )
    df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "importance_mean_minus_std": result.importances_mean
            - result.importances_std,
        }
    )
    priority = {f: i for i, f in enumerate(DEFAULT_EVENT_PREDICTORS)}
    available_priority = df["feature"].map(priority).fillna(len(DEFAULT_EVENT_PREDICTORS))
    return df.assign(_priority=available_priority).sort_values(
        ["_priority", "importance_mean"], ascending=[True, False]
    ).drop(columns=["_priority"]).reset_index(drop=True)
