from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.compose import ColumnTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import TORCH_METADATA_PATH, TORCH_MODEL_PATH
from src.utils.io import hash_dataframe, save_json


class TelcoMLP(nn.Module):
    
    #MLP for churn prediction using preprocessed tabular features built with pytorch
    

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_mlp(input_dim: int) -> TelcoMLP:
    return TelcoMLP(input_dim)


def _to_dense_array(X) -> np.ndarray:
    """
    Converts sparse matrix to a dense numpy array.

    The Telco dataset is small enough that calling .toarray() is acceptable.
    """

    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    features = torch.as_tensor(X, dtype=torch.float32)
    targets = torch.as_tensor(y, dtype=torch.float32)
    ds = TensorDataset(features, targets)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    # y is 1D array of 0/1
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)


@dataclass
class TorchTrainingResult:
    model: TelcoMLP
    metadata: Dict


def train_torch_model(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 5,
) -> TorchTrainingResult:
    """
    Training phase of PyTorch MLP on preprocessed features with early stopping on val AUC.
    """

    # Transform data with fitted preprocessor
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    X_train_arr = _to_dense_array(X_train_proc)
    X_val_arr = _to_dense_array(X_val_proc)

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()

    input_dim = X_train_arr.shape[1]
    model = build_mlp(input_dim)

    pos_weight = _compute_pos_weight(y_train_arr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = _make_dataloader(X_train_arr, y_train_arr, batch_size=batch_size, shuffle=True)
    val_loader = _make_dataloader(X_val_arr, y_val_arr, batch_size=batch_size, shuffle=False)

    best_val_auc = -np.inf
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        all_val_logits = []
        all_val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                all_val_logits.append(logits.cpu().numpy())
                all_val_targets.append(yb.cpu().numpy())

        val_logits = np.concatenate(all_val_logits)
        val_targets = np.concatenate(all_val_targets)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        # compute AUC
        from sklearn.metrics import roc_auc_score

        val_auc = roc_auc_score(val_targets, val_probs)

        print(f"[Torch] Epoch {epoch:03d} | loss={epoch_loss:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    metadata: Dict = {
        "input_dim": int(input_dim),
        "best_val_auc": float(best_val_auc),
        "n_epochs_trained": int(epoch),
        "train_timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_hash": hash_dataframe(X_train),
    }

    return TorchTrainingResult(model=model, metadata=metadata)


def save_torch_artifacts(result: TorchTrainingResult) -> None:
    TORCH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TORCH_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(result.model.state_dict(), TORCH_MODEL_PATH)
    save_json(result.metadata, TORCH_METADATA_PATH)


def predict_proba_torch(model: TelcoMLP, X_proc) -> np.ndarray:
    
    #Run inference with the trained PyTorch model on preprocessed features.
    

    X_arr = _to_dense_array(X_proc)
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X_arr, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


if __name__ == "__main__":
    # Simple manual training demo: use 80/20 split with val==test here.
    from sklearn.model_selection import train_test_split

    from src.data.load_data import load_telco_data
    from src.features.preprocessing import build_preprocessor

    X, y = load_telco_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preproc = build_preprocessor(X_train)
    preproc.fit(X_train)

    result = train_torch_model(preproc, X_train, y_train, X_val, y_val)
    save_torch_artifacts(result)

