"""Feature selection utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


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
    corr_threshold: float = 0.9,
    relevance_threshold: float = 0.05,
) -> list[str]:
    """Select relevant features while removing multicollinearity.

    The function first removes features that are highly correlated
    with each other and then keeps only those features that have
    at least ``relevance_threshold`` absolute correlation with the target.
    """
    df = df.dropna(subset=[target_col])
    features = df.drop(columns=[target_col])
    features = remove_multicollinearity(features, threshold=corr_threshold)
    target_corr = df[features.columns].corrwith(df[target_col]).abs()
    selected = list(target_corr[target_corr >= relevance_threshold].index)
    return selected


def select_features_rf_cv(
    df: pd.DataFrame,
    target_col: str,
    max_features: int | None = None,
    cv: int = 3,
    corr_threshold: float = 0.9,
    random_state: int = 42,
) -> list[str]:
    """Select features using RandomForest feature importance with CV.

    The top ``max_features`` according to the averaged feature importance are
    kept and then filtered for multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features and the target.
    target_col : str
        Name of the target column in ``df``.
    max_features : int, optional
        Maximum number of features to keep before removing multicollinearity.
        Defaults to ``sqrt(n_features) + 7``.
    cv : int, optional
        Number of time series splits. Defaults to ``3``.
    corr_threshold : float, optional
        Threshold to remove correlated features after ranking. Defaults to
        ``0.9``.
    random_state : int, optional
        Random state for ``RandomForestRegressor``. Defaults to ``42``.
    """

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit

    df = df.dropna(subset=[target_col])
    features = df.drop(columns=[target_col])
    y = df[target_col]

    if features.empty:
        return []

    n_total = features.shape[1]
    if max_features is None:
        max_features = int(np.sqrt(n_total)) + 7

    splits = min(cv, max(1, len(df) - 1))
    tscv = TimeSeriesSplit(n_splits=splits)
    importances = np.zeros(n_total)
    for train_idx, _ in tscv.split(features):
        model = RandomForestRegressor(random_state=random_state)
        model.fit(features.iloc[train_idx], y.iloc[train_idx])
        importances += model.feature_importances_
    importances /= tscv.get_n_splits()

    order = np.argsort(importances)[::-1]
    top_cols = features.columns[order[:max_features]]
    filtered = remove_multicollinearity(features[top_cols], threshold=corr_threshold)
    return list(filtered.columns)
