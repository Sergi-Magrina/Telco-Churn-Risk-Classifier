from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _get_numeric_and_categorical_columns(X: pd.DataFrame) -> tuple[List[str], List[str]]:
    #infers numeric and categorical columns from the input dataFrame.

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    builds a ColumnTransformer-based preprocessing pipeline for the Telco dataset.

    Numeric columns:
        - SimpleImputer(strategy=median)
        - StandardScaler

    Categorical columns:
        - SimpleImputer(strategy=most_frequent)
        - OneHotEncoder(handle_unknown=ignore, sparse=True)

    IMPORTANT: The returned preprocessor is only fitted only on training data
    (X_train) to avoid test set leakage.
    """

    num_cols, cat_cols = _get_numeric_and_categorical_columns(X)

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    return preprocessor


if __name__ == "__main__":
    #manual check when running directly
    from src.data.load_data import load_telco_data
    from src.data.split import train_test_split_leakage_safe

    X, y = load_telco_data()
    X_train, X_test, y_train, y_test = train_test_split_leakage_safe(X, y)

    preproc = build_preprocessor(X_train)
    X_train_proc = preproc.fit_transform(X_train)
    X_test_proc = preproc.transform(X_test)

    print("Preprocessor built.")
    print("X_train_proc shape:", X_train_proc.shape)
    print("X_test_proc shape:", X_test_proc.shape)
    print("Type of transformed data:", type(X_train_proc))

