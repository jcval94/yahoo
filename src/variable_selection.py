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
