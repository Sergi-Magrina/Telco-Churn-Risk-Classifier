from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from urllib.request import urlopen

import pandas as pd
import torch
from fastapi import Body, FastAPI, Query
from joblib import load as joblib_load

from src.config import (
    SKLEARN_METADATA_PATH,
    SKLEARN_MODEL_PATH,
    TORCH_METADATA_PATH,
    TORCH_MODEL_PATH,
    TORCH_PREPROCESSOR_PATH,
)
from src.modeling.torch_model import TelcoMLP, predict_proba_torch
from src.serving.schemas import CustomerInput
from src.utils.io import load_json

app = FastAPI(title="Telco Churn Risk API")

# Loaded once at startup
SKLEARN_PIPELINE = None
TORCH_MODEL = None
TORCH_PREPROCESSOR = None

# Request-body example shown in Swagger UI (/docs)
EXAMPLE_CUSTOMER: Dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def _ensure_file(local_path: Path, url_env_key: str) -> None:
    """
    Ensure `local_path` exists. If missing, download it from the URL stored in env var `url_env_key`.
    This is required on Render because the instance filesystem won't contain locally-trained artifacts.
    """
    if local_path.exists():
        return

    url = os.getenv(url_env_key)
    if not url:
        raise RuntimeError(
            f"Missing required file: {local_path}. "
            f"Set env var {url_env_key} to a direct download URL "
            f"(e.g. Hugging Face /resolve/main/... link)."
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[startup] Downloading {url_env_key} -> {local_path}")
    with urlopen(url) as r, open(local_path, "wb") as f:
        f.write(r.read())


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

    p = predict_proba_torch(TORCH_MODEL, X_proc)[0]
    return float(p)


@app.on_event("startup")
def startup_load_artifacts() -> None:
    global SKLEARN_PIPELINE, TORCH_MODEL, TORCH_PREPROCESSOR

    # --- Download missing artifacts (Render) ---
    _ensure_file(SKLEARN_MODEL_PATH, "SKLEARN_MODEL_URL")
    _ensure_file(SKLEARN_METADATA_PATH, "SKLEARN_META_URL")

    _ensure_file(TORCH_MODEL_PATH, "TORCH_MODEL_URL")
    _ensure_file(TORCH_PREPROCESSOR_PATH, "TORCH_PREP_URL")
    _ensure_file(TORCH_METADATA_PATH, "TORCH_META_URL")

    # --- Load sklearn ---
    SKLEARN_PIPELINE = joblib_load(SKLEARN_MODEL_PATH)

    # --- Load torch preprocessor + model ---
    TORCH_PREPROCESSOR = joblib_load(TORCH_PREPROCESSOR_PATH)

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
    payload: CustomerInput = Body(
        ...,
        openapi_examples={
            "demo": {"summary": "Demo customer", "value": EXAMPLE_CUSTOMER}
        },
    ),
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