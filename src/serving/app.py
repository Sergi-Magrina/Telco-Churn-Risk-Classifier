from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import torch
from fastapi import FastAPI, Query
from joblib import load as joblib_load

from src.config import (
    SKLEARN_MODEL_PATH,
    TORCH_METADATA_PATH,
    TORCH_MODEL_PATH,
    TORCH_PREPROCESSOR_PATH,
)
from src.modeling.torch_model import TelcoMLP, predict_proba_torch
from src.utils.io import load_json
from src.serving.schemas import CustomerInput


app = FastAPI(title="Telco Churn Risk API")

# Loaded once at startup
SKLEARN_PIPELINE = None
TORCH_MODEL = None
TORCH_PREPROCESSOR = None


def _payload_to_df(payload: CustomerInput) -> pd.DataFrame:
    # pydantic v2 uses model_dump; v1 uses dict
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    return pd.DataFrame([data])


def _bucket(p: float) -> str:
    if p < 0.30:
        return "low"
    if p < 0.60:
        return "medium"
    return "high"


def _predict_sklearn(df: pd.DataFrame) -> float:
    proba = SKLEARN_PIPELINE.predict_proba(df)[:, 1][0]
    return float(proba)


def _predict_torch(df: pd.DataFrame) -> float:
    X_proc = TORCH_PREPROCESSOR.transform(df)

    # If sparse matrix, convert to dense for torch
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    # predict_proba_torch will return an array-like of probabilities
    p = predict_proba_torch(TORCH_MODEL, X_proc)[0]
    return float(p)


@app.on_event("startup")
def startup_load_artifacts() -> None:
    global SKLEARN_PIPELINE, TORCH_MODEL, TORCH_PREPROCESSOR

    # 1) sklearn pipeline already includes preprocessing
    SKLEARN_PIPELINE = joblib_load(SKLEARN_MODEL_PATH)

    # 2) torch preprocessor must be the fitted one saved during training
    TORCH_PREPROCESSOR = joblib_load(TORCH_PREPROCESSOR_PATH)

    # 3) torch model
    meta = load_json(TORCH_METADATA_PATH)
    input_dim = int(meta["input_dim"])

    TORCH_MODEL = TelcoMLP(input_dim=input_dim)
    state_dict = torch.load(TORCH_MODEL_PATH, map_location="cpu")
    TORCH_MODEL.load_state_dict(state_dict)
    TORCH_MODEL.eval()

    print("API startup: loaded sklearn pipeline + torch model + torch preprocessor")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "sklearn_loaded": SKLEARN_PIPELINE is not None,
        "torch_loaded": TORCH_MODEL is not None,
        "torch_preprocessor_loaded": TORCH_PREPROCESSOR is not None,
    }


@app.get("/")
def root():
    return {"message": "Telco Churn Risk API is running. Visit /docs"}


@app.post("/predict")
def predict(
    payload: CustomerInput,
    backend: str = Query("sklearn", pattern="^(sklearn|torch|both)$"),
) -> Dict[str, Any]:
    df = _payload_to_df(payload)

    if backend == "sklearn":
        p = _predict_sklearn(df)
        return {
            "model_type": "sklearn",
            "churn_probability": p,
            "risk_bucket": _bucket(p),
        }

    if backend == "torch":
        p = _predict_torch(df)
        return {
            "model_type": "torch",
            "churn_probability": p,
            "risk_bucket": _bucket(p),
        }

    # backend == "both"
    p_s = _predict_sklearn(df)
    p_t = _predict_torch(df)
    return {
        "model_type": "both",
        "sklearn": {"churn_probability": p_s, "risk_bucket": _bucket(p_s)},
        "torch": {"churn_probability": p_t, "risk_bucket": _bucket(p_t)},
        "delta_torch_minus_sklearn": p_t - p_s,
    }
