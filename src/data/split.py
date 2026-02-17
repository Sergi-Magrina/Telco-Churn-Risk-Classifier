from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_leakage_safe(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    leakage-safe train/test split on the cleaned Telco dataset.

    This function must be called on raw cleaned features and target, before any
    preprocessing (imputation, scaling, encoding, ...) is done, to avoid
    leaking information from the test set into the training pipeline (important)
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Small sanity check CLI: load data via load_data and run split.
    from src.data.load_data import load_telco_data

    X_all, y_all = load_telco_data()
    X_tr, X_te, y_tr, y_te = train_test_split_leakage_safe(X_all, y_all)

    print("Train shape:", X_tr.shape, "Test shape:", X_te.shape)
    print("Train churn rate:", y_tr.mean(), "Test churn rate:", y_te.mean())

