from __future__ import annotations

from joblib import load
import numpy as np

from src.config import (
    COMPARISON_REPORT_PATH,
    METRICS_PATH,
    SKLEARN_MODEL_PATH,
    TORCH_METADATA_PATH,
    TORCH_MODEL_PATH,
)
from src.data.load_data import load_telco_data
from src.data.split import train_test_split_leakage_safe
from src.features.preprocessing import build_preprocessor
from src.modeling.evaluate import evaluate_single_model
from src.modeling.torch_model import TelcoMLP, predict_proba_torch
from src.utils.io import load_json, save_json
from src.utils.seed import set_global_seed


def load_torch_model(input_dim: int) -> TelcoMLP:
    import torch

    model = TelcoMLP(input_dim=input_dim)
    state_dict = torch.load(TORCH_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()  
    return model



def main() -> None:
    set_global_seed()

    X, y = load_telco_data()
    X_train, X_test, y_train, y_test = train_test_split_leakage_safe(X, y)

    # sklearn model (full pipeline)
    sklearn_model = load(SKLEARN_MODEL_PATH)

    # rebuild preprocessor for torch model on training data to ensure same features are bein used
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    torch_meta = load_json(TORCH_METADATA_PATH)
    input_dim = torch_meta["input_dim"]

    torch_model = load_torch_model(input_dim)

    # wrap torch model into a callable that accepts preprocessed X
    def torch_predict(X_proc):
        return predict_proba_torch(torch_model, X_proc)

    sklearn_metrics = evaluate_single_model(
        name="sklearn",
        model=sklearn_model,
        X_test=X_test,
        y_test=y_test,
    )

    torch_metrics = evaluate_single_model(
        name="torch",
        model=lambda X_proc: torch_predict(X_proc),
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
    )

    metrics = {
        "sklearn": sklearn_metrics,
        "torch": torch_metrics,
    }
    save_json(metrics, METRICS_PATH)

    # the comparison markdown (can be found in reports/comparison.md)
    lines = []
    lines.append("# Model Comparison")
    lines.append("")
    lines.append("| Model | ROC-AUC | PR-AUC | Brier | Latency (ms/run on 1000 rows) |")
    lines.append("|-------|--------:|-------:|------:|-------------------------------:|")
    for name, m in metrics.items():
        roc_auc = m["classification"]["roc_auc"]
        pr_auc = m["classification"]["pr_auc"]
        brier = m["calibration"]["brier_score"]
        latency = m["latency_ms_per_run_on_sample"]
        lines.append(f"| {name} | {roc_auc:.4f} | {pr_auc:.4f} | {brier:.4f} | {latency:.2f} |")

    # recommendation: prefer sklearn unless torch is substantially better
    better_model = "sklearn"
    if metrics["torch"]["classification"]["roc_auc"] > metrics["sklearn"]["classification"]["roc_auc"] + 0.01:
        better_model = "torch"

    lines.append("")
    lines.append(f"Recommended model to ship: **{better_model}** based on ROC-AUC, PR-AUC, calibration, and latency.")

    COMPARISON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("comparison report written", COMPARISON_REPORT_PATH)


if __name__ == "__main__":
    main()

