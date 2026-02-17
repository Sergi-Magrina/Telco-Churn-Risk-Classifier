from __future__ import annotations

import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import roc_curve, auc
import torch
import numpy as np

from src.config import (
    SKLEARN_MODEL_PATH,
    TORCH_MODEL_PATH,
    TORCH_METADATA_PATH,
    TORCH_PREPROCESSOR_PATH,
)
from src.data.load_data import load_telco_data
from src.data.split import train_test_split_leakage_safe
from src.modeling.torch_model import TelcoMLP, predict_proba_torch
from src.utils.io import load_json


def main() -> None:
    # Load data
    X, y = load_telco_data()
    X_train, X_test, y_train, y_test = train_test_split_leakage_safe(X, y)

    # ---- sklearn ----
    sklearn_model = load(SKLEARN_MODEL_PATH)
    y_prob_sklearn = sklearn_model.predict_proba(X_test)[:, 1]

    fpr_s, tpr_s, _ = roc_curve(y_test, y_prob_sklearn)
    roc_auc_s = auc(fpr_s, tpr_s)

    # ---- torch ----
    torch_meta = load_json(TORCH_METADATA_PATH)
    input_dim = torch_meta["input_dim"]

    torch_model = TelcoMLP(input_dim=input_dim)
    state_dict = torch.load(TORCH_MODEL_PATH, map_location="cpu")
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    preprocessor = load(TORCH_PREPROCESSOR_PATH)
    X_test_proc = preprocessor.transform(X_test)

    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    y_prob_torch = predict_proba_torch(torch_model, X_test_proc)

    fpr_t, tpr_t, _ = roc_curve(y_test, y_prob_torch)
    roc_auc_t = auc(fpr_t, tpr_t)

    # ---- plot ----
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_s, tpr_s, label=f"Sklearn LR (AUC = {roc_auc_s:.4f})")
    plt.plot(fpr_t, tpr_t, label=f"Torch MLP (AUC = {roc_auc_t:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=300)
    plt.close()

    print("ROC curve saved to reports/roc_curve.png")


if __name__ == "__main__":
    main()
