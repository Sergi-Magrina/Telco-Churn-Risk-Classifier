from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import RAW_TELCO_PATH


def load_telco_data(path: str | Path = RAW_TELCO_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    goal of func: to load and clean the Telco customer churn dataset.

    Steps:
    - Read the CSV from the given path (by default `data/raw/telco.csv` using `RAW_TELCO_PATH`)
    - Drop the `customerID` column (not a predictive feat.)
    - Coerce `TotalCharges` to numeric with `errors="coerce"`
    - Drop rows where `TotalCharges` is NaN
    - Convert the target `Churn` from "Yes", "No" to integers 1, 0
    - Split into features `X` and target `y` and return them

    Parameters
    ----------
    path:
        Path that points to the raw Telco CSV in `/data/raw/telco.csv` 

    Returns
    -------
    X : pd.DataFrame
        cleaned feature matrix with `Churn` removed.
    y : pd.Series
        binary target series where 1 indicates churn.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Telco CSV not found at {csv_path}. "
        )

    df = pd.read_csv(csv_path)

    # drop identifier column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # coerce TotalCharges to numeric and drop rows with invalid/missing values (NaN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        before_drop = len(df)
        df = df.dropna(subset=["TotalCharges"])
        after_drop = len(df)
        dropped = before_drop - after_drop
        if dropped > 0:
            # (no logger of dropped rows yet. Dropped rows found to be 11)
            print(f"Dropped {dropped} rows due to invalid `TotalCharges`.")

    # map target column Churn from {'Yes', 'No'} to {1, 0}
    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' not found in dataset.")

    churn_mapping = {"Yes": 1, "No": 0}
    df["Churn"] = df["Churn"].map(churn_mapping)

    if df["Churn"].isna().any():
        raise ValueError(
            "Some values in 'Churn' could not be mapped to {1, 0}. "
            "Expected values: 'Yes' or 'No'."
        )

    # separate features and target columns
    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    return X, y


def _print_diagnostics(X: pd.DataFrame, y: pd.Series) -> None:
    """
    this prints basic dataset diagnostics

    - Shape of features and basic info about target.
    - Class balance / churn rate.
    - Missing value counts per column.
    """

    print("--Telco Churn Dataset Diagnostics--")
    print(f"Features shape: {X.shape}")
    print(f"Target length: {len(y)}")

    churn_rate = y.mean()
    print(f"Churn rate (mean of target): {churn_rate:.4f}")
    print("Class distribution (0 = no churn, 1 = churn):")
    print(y.value_counts(normalize=False).sort_index())
    print("Class distribution (proportions):")
    print(y.value_counts(normalize=True).sort_index())

    print("\nMissing values per feature column:")
    missing = X.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0] if (missing > 0).any() else "No missing values in features.")


if __name__ == "__main__":
    # Simple CLI / demo usage:
    # Load data from the default location and print diagnostics.
    X_, y_ = load_telco_data()
    _print_diagnostics(X_, y_)

