from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_openml


def load_data(data_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset from CSV if present, otherwise download and persist it."""
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        dataset = fetch_openml(name="credit-g", version=1, as_frame=True)
        df = dataset.frame.copy()
        df.to_csv(data_path, index=False)

    target_col = "class"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in dataset.")

    X = df.drop(columns=[target_col]).copy()
    y = (df[target_col].astype(str).str.lower() == "bad").astype(int)
    y.name = "is_bad_credit"
    return X, y
