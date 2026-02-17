from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent_dir(path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def hash_dataframe(df: pd.DataFrame) -> str:

    df_sorted = df.sort_index(axis=1)
    csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()

