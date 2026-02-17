from __future__ import annotations

from fastapi.testclient import TestClient
from joblib import load
import numpy as np

from src.config import SKLEARN_MODEL_PATH
from src.data.load_data import load_telco_data
from src.serving.app import app


def _build_example_payload():
    X, y = load_telco_data()
    sample = X.iloc[0].to_dict()
    return sample


def test_sklearn_pipeline_inference_shape():
    model = load(SKLEARN_MODEL_PATH)
    X, y = load_telco_data()
    probs = model.predict_proba(X.iloc[:5])[:, 1]
    assert probs.shape == (5,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_api_predict_contract():
    client = TestClient(app)
    payload = _build_example_payload()
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "churn_probability" in data
    assert "risk_bucket" in data
    assert "model_type" in data

