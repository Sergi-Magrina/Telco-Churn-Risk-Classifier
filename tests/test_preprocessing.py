from __future__ import annotations

from src.data.load_data import load_telco_data
from src.data.split import train_test_split_leakage_safe
from src.features.preprocessing import build_preprocessor


def test_preprocessing_fit_transform_and_transform():
    X, y = load_telco_data()
    X_train, X_test, y_train, y_test = train_test_split_leakage_safe(X, y)

    preproc = build_preprocessor(X_train)
    X_train_proc = preproc.fit_transform(X_train)
    X_test_proc = preproc.transform(X_test)

    assert X_train_proc.shape[0] == X_train.shape[0]
    assert X_test_proc.shape[0] == X_test.shape[0]

    # Ensure output is sparse-like with same number of rows
    from scipy import sparse

    assert sparse.issparse(X_train_proc)
    assert sparse.issparse(X_test_proc)

