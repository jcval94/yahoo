"""Feature selection utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit


def remove_multicollinearity(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Return DataFrame with highly correlated columns removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature matrix.
    threshold : float, optional
        Absolute correlation above which one of a pair of columns is dropped.
        Defaults to ``0.9``.
    """
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)


def select_features(
    df: pd.DataFrame,
    target_col: str,
    *,
    sample_size: int | None = 256,
    n_splits: int = 3,
    corr_threshold: float = 0.9,
) -> list[str]:
    """Select features using CV permutation importance and multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with features and target.
    target_col : str
        Name of the target column.
    sample_size : int or None, optional
        Number of most recent rows to use for the selection. ``None`` uses all
        rows. Defaults to ``256`` for faster execution on large datasets.
    n_splits : int, optional
        Number of CV splits for ``TimeSeriesSplit``. Defaults to ``3``.
    corr_threshold : float, optional
        Absolute correlation above which a candidate feature is dropped when
        another selected feature is already present. Defaults to ``0.9``.

    Returns
    -------
    list[str]
        Ordered list of selected feature names.

    Notes
    -----
    The number of returned features is capped at ``sqrt(n_vars) + 7`` where
    ``n_vars`` is the original feature count. Permutation importance is
    computed using a ``RandomForestRegressor`` on a rolling ``TimeSeriesSplit``.
    Multicollinearity is used as a secondary criterion when selecting the most
    important features.
    """

    df = df.dropna(subset=[target_col])
    if df.empty:
        return []

    if sample_size is not None and len(df) > sample_size:
        df = df.tail(sample_size)

    features = df.drop(columns=[target_col])
    y = df[target_col]

    n_splits = min(n_splits, max(1, len(df) - 1))
    cv = TimeSeriesSplit(n_splits=n_splits)
    importances = np.zeros(features.shape[1])

    for train_idx, test_idx in cv.split(features):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            model,
            features.iloc[test_idx],
            y.iloc[test_idx],
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        importances += result.importances_mean
    importances /= cv.get_n_splits()

    rankings = pd.Series(importances, index=features.columns).sort_values(
        ascending=False
    )

    max_features = int(np.sqrt(len(rankings))) + 7
    corr = features.corr().abs()
    selected: list[str] = []
    for feat in rankings.index:
        if len(selected) >= max_features:
            break
        if all(corr.at[feat, s] <= corr_threshold for s in selected):
            selected.append(feat)

    return selected
