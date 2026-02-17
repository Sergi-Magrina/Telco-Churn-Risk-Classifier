from __future__ import annotations

from src.config import RAW_TELCO_PATH
from src.data.load_data import load_telco_data
from src.data.split import train_test_split_leakage_safe
from src.features.preprocessing import build_preprocessor
from src.modeling.sklearn_model import save_sklearn_artifacts, train_sklearn_model
from src.utils.seed import set_global_seed


def main() -> None:
    set_global_seed()

    print(f"Loading Telco data from {RAW_TELCO_PATH} ...")
    X, y = load_telco_data()

    X_train, X_test, y_train, y_test = train_test_split_leakage_safe(X, y)

    preprocessor = build_preprocessor(X_train)

    print("Training sklearn LogisticRegression with GridSearchCV...")
    model, metadata = train_sklearn_model(X_train, y_train, preprocessor)

    print("Saving sklearn model and metadata...")
    save_sklearn_artifacts(model, metadata)
    print("Done.")


if __name__ == "__main__":
    main()

