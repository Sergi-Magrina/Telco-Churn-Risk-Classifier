from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import SKLEARN_METADATA_PATH, SKLEARN_MODEL_PATH
from src.utils.io import hash_dataframe, save_json


def build_sklearn_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    
    #builds the sklearn Pipeline where preprocessing is chained with the LogisticRegression classifier.
    

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    return pipeline


def train_sklearn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Tuple[Pipeline, Dict]:
    """
    Trains the sklearn baseline model with hyperparameter tuning.

    Uses GridSearchCV over C for LogisticRegression and optimizes ROC-AUC.
    """

    pipeline = build_sklearn_pipeline(preprocessor)

    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    grid_search.fit(X_train, y_train)

    best_pipeline: Pipeline = grid_search.best_estimator_

    # computes feature count after preprocessing (will be larger because of onehotencoder)
    X_encoded = best_pipeline.named_steps["preprocess"].fit_transform(X_train)
    feature_count = X_encoded.shape[1]

    # Basic training ROC-AUC for reference
    train_probs = best_pipeline.predict_proba(X_train)[:, 1]
    train_auc = float(roc_auc_score(y_train, train_probs))

    metadata: Dict = {
        "feature_count": int(feature_count),
        "best_params": grid_search.best_params_,
        "best_cv_score_roc_auc": float(grid_search.best_score_),
        "train_roc_auc": train_auc,
        "train_timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_hash": hash_dataframe(X_train),
    }

    return best_pipeline, metadata


def save_sklearn_artifacts(model: Pipeline, metadata: Dict) -> None:
    
    #saves the trained sklearn pipeline and metadata to disk.
    

    SKLEARN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SKLEARN_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    dump(model, SKLEARN_MODEL_PATH)
    save_json(metadata, SKLEARN_METADATA_PATH)


if __name__ == "__main__":
    # small manual training run when executed directly.
    from src.data.load_data import load_telco_data
    from src.data.split import train_test_split_leakage_safe
    from src.features.preprocessing import build_preprocessor

    X, y = load_telco_data()
    X_train, X_test, y_train, y_test = train_test_split_leakage_safe(X, y)

    preproc = build_preprocessor(X_train)
    model, meta = train_sklearn_model(X_train, y_train, preproc)
    save_sklearn_artifacts(model, meta)

    y_test_probs = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_probs)
    print("Sklearn LogisticRegression test ROC-AUC:", test_auc)

