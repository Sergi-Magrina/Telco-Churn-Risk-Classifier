from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.utils.timing import benchmark_inference


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    
    #Computes standard classification metrics for the classification.
    

    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    y_pred = (y_prob >= threshold).astype(int)
    precision_t, recall_t, f1_t, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_at_threshold": float(precision_t),
        "recall_at_threshold": float(recall_t),
        "f1_at_threshold": float(f1_t),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    
    #calibration curve points and Brier score.
    

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "n_bins": int(n_bins),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "brier_score": float(brier),
    }


def evaluate_single_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer | None = None,
    latency_sample_size: int = 1000,
) -> Dict:
    """
    Evaluates a model (sklearn pipeline or torch) on the test set.

    For sklearn, `model` is a Pipeline with predict_proba.
    For torch, `model` exposes a callable returning probabilities given
    preprocessed features from preprocessor.joblib
    """

    y_true = y_test.to_numpy()

    if isinstance(model, Pipeline):
        # sklearn pipeline: includes preprocessing
        y_prob = model.predict_proba(X_test)[:, 1]

        # For latency, use raw X_test and rely on the pipeline
        latency_ms = benchmark_inference(
            func=lambda X: model.predict_proba(X),
            X=X_test.iloc[:latency_sample_size],
            n_runs=5,
        )
    else:
        if preprocessor is None:
            raise ValueError("preprocessor must be provided when evaluating a non-sklearn model.")
        X_test_proc = preprocessor.transform(X_test)
        y_prob = model(X_test_proc)
        y_prob = np.asarray(y_prob).reshape(-1)

        latency_ms = benchmark_inference(
            func=lambda X: model(preprocessor.transform(X)),
            X=X_test.iloc[:latency_sample_size],
            n_runs=5,
        )

    cls_metrics = compute_classification_metrics(y_true, y_prob)
    calib_metrics = compute_calibration(y_true, y_prob)

    return {
        "name": name,
        "classification": cls_metrics,
        "calibration": calib_metrics,
        "latency_ms_per_run_on_sample": float(latency_ms),
    }

