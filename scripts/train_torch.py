from __future__ import annotations

from sklearn.model_selection import train_test_split

from src.config import RAW_TELCO_PATH
from src.data.load_data import load_telco_data
from src.features.preprocessing import build_preprocessor
from src.modeling.torch_model import save_torch_artifacts, train_torch_model
from src.utils.seed import set_global_seed
from joblib import dump
from src.config import TORCH_PREPROCESSOR_PATH


def main() -> None:
    set_global_seed()

    print(f"Loading Telco data from {RAW_TELCO_PATH} ...")
    X, y = load_telco_data()

    # For torch we create a train/val split from the cleaned data.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    TORCH_PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(preprocessor, TORCH_PREPROCESSOR_PATH)
    print(f"Saved torch preprocessor to {TORCH_PREPROCESSOR_PATH}")

    print("Training PyTorch MLP...")
    result = train_torch_model(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    print("Saving torch model and metadata...")
    save_torch_artifacts(result)
    print("Done.")


if __name__ == "__main__":
    main()

