"""
Porto Seguro Safe Driver Prediction - data loading and preprocessing.

Schema inference is suffix-based:
  _cat  -> categorical
  _bin  -> binary
  other (excluding 'target', 'id') -> numeric
  target -> label
  id     -> excluded
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

def infer_schema(columns: List[str]) -> Dict[str, str]:
    """Return a dict mapping each column name to its type.

    Types: 'numeric', 'categorical', 'binary', 'target', 'id', 'excluded'.
    """
    schema: Dict[str, str] = {}
    for col in columns:
        if col == "target":
            schema[col] = "target"
        elif col == "id":
            schema[col] = "id"
        elif col.endswith("_cat"):
            schema[col] = "categorical"
        elif col.endswith("_bin"):
            schema[col] = "binary"
        else:
            schema[col] = "numeric"
    return schema


# ---------------------------------------------------------------------------
# Numeric statistics (observation-only, ignoring -1)
# ---------------------------------------------------------------------------

def compute_numeric_stats(
    df: pd.DataFrame,
    numeric_cols: List[str],
    missing_value: float = -1.0,
) -> Dict[str, Dict[str, float]]:
    """Compute mean and std for each numeric column using observed values only.

    Values equal to ``missing_value`` are treated as missing.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        observed = df.loc[df[col] != missing_value, col]
        mean = float(observed.mean()) if len(observed) > 0 else 0.0
        std = float(observed.std()) if len(observed) > 1 else 1.0
        if std == 0.0 or np.isnan(std):
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


# ---------------------------------------------------------------------------
# Categorical mappings
# ---------------------------------------------------------------------------

def build_categorical_mappings(
    df: pd.DataFrame,
    cat_cols: List[str],
    missing_value: int = -1,
) -> Dict[str, Dict[int, int]]:
    """Build value -> integer-ID mapping for each categorical column.

    ``missing_value`` (-1) is mapped to a dedicated ID (0 by convention).
    Other values start at 1.
    """
    mappings: Dict[str, Dict[int, int]] = {}
    for col in cat_cols:
        values = sorted(df[col].unique())
        mapping: Dict[int, int] = {}
        next_id = 1
        for v in values:
            if int(v) == missing_value:
                mapping[int(v)] = 0  # reserved "unknown" ID
            else:
                mapping[int(v)] = next_id
                next_id += 1
        mappings[col] = mapping
    return mappings


# ---------------------------------------------------------------------------
# Binary mappings
# ---------------------------------------------------------------------------

def build_binary_mappings(
    df: pd.DataFrame,
    bin_cols: List[str],
    missing_value: int = -1,
) -> Dict[str, Dict[int, int]]:
    """Build value -> integer-ID mapping for binary columns.

    Values: 0 -> 1, 1 -> 2, -1 (missing) -> 0.
    This keeps 0 as the "unknown" index consistent with categorical.
    """
    mappings: Dict[str, Dict[int, int]] = {}
    for col in bin_cols:
        values = sorted(df[col].unique())
        mapping: Dict[int, int] = {}
        next_id = 1
        for v in sorted(v for v in values if int(v) != missing_value):
            mapping[int(v)] = next_id
            next_id += 1
        mapping[missing_value] = 0
        mappings[col] = mapping
    return mappings


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class PortoPreprocessor:
    """Fit on training data, transform train/valid/test splits."""

    def __init__(self, missing_value: float = -1.0) -> None:
        self.missing_value = missing_value
        self.schema: Dict[str, str] = {}
        self.numeric_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.bin_cols: List[str] = []
        self.numeric_stats: Dict[str, Dict[str, float]] = {}
        self.cat_mappings: Dict[str, Dict[int, int]] = {}
        self.bin_mappings: Dict[str, Dict[int, int]] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PortoPreprocessor":
        self.schema = infer_schema(list(df.columns))
        self.numeric_cols = [c for c, t in self.schema.items() if t == "numeric"]
        self.cat_cols = [c for c, t in self.schema.items() if t == "categorical"]
        self.bin_cols = [c for c, t in self.schema.items() if t == "binary"]
        self.numeric_stats = compute_numeric_stats(df, self.numeric_cols, self.missing_value)
        self.cat_mappings = build_categorical_mappings(df, self.cat_cols, int(self.missing_value))
        self.bin_mappings = build_binary_mappings(df, self.bin_cols, int(self.missing_value))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Return a dict of tensors ready for the model.

        Keys:
          x_num         float32 (N, num_features)   - mean-imputed, standardised
          missing_mask  bool    (N, num_features)    - True where original value == missing
          x_cat         long    (N, num_cat_features)
          x_bin         long    (N, num_bin_features)
          y             float32 (N,)                 - only present when 'target' in df
        """
        N = len(df)
        result: Dict[str, torch.Tensor] = {}

        # ---- numeric ----
        if self.numeric_cols:
            x_num = np.zeros((N, len(self.numeric_cols)), dtype=np.float32)
            missing_mask = np.zeros((N, len(self.numeric_cols)), dtype=bool)
            for i, col in enumerate(self.numeric_cols):
                raw = df[col].to_numpy(dtype=np.float32)
                is_missing = raw == self.missing_value
                missing_mask[:, i] = is_missing
                mean = float(self.numeric_stats[col]["mean"])
                std = float(self.numeric_stats[col]["std"])
                observed = np.where(is_missing, mean, raw)
                x_num[:, i] = (observed - mean) / std
            result["x_num"] = torch.tensor(x_num)
            result["missing_mask"] = torch.tensor(missing_mask)
        else:
            result["x_num"] = torch.zeros((N, 0), dtype=torch.float32)
            result["missing_mask"] = torch.zeros((N, 0), dtype=torch.bool)

        # ---- categorical ----
        if self.cat_cols:
            x_cat = np.zeros((N, len(self.cat_cols)), dtype=np.int64)
            for i, col in enumerate(self.cat_cols):
                mapping = self.cat_mappings[col]
                raw = df[col].to_numpy(dtype=np.int64)
                # unseen values -> 0
                x_cat[:, i] = np.vectorize(lambda v: mapping.get(int(v), 0))(raw)
            result["x_cat"] = torch.tensor(x_cat)
        else:
            result["x_cat"] = torch.zeros((N, 0), dtype=torch.long)

        # ---- binary ----
        if self.bin_cols:
            x_bin = np.zeros((N, len(self.bin_cols)), dtype=np.int64)
            for i, col in enumerate(self.bin_cols):
                mapping = self.bin_mappings[col]
                raw = df[col].to_numpy(dtype=np.int64)
                x_bin[:, i] = np.vectorize(lambda v: mapping.get(int(v), 0))(raw)
            result["x_bin"] = torch.tensor(x_bin)
        else:
            result["x_bin"] = torch.zeros((N, 0), dtype=torch.long)

        # ---- target ----
        if "target" in df.columns:
            result["y"] = torch.tensor(df["target"].to_numpy(dtype=np.float32))

        return result

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        return self.fit(df).transform(df)

    # convenience properties for model construction
    @property
    def num_numeric(self) -> int:
        return len(self.numeric_cols)

    @property
    def num_cat(self) -> int:
        return len(self.cat_cols)

    @property
    def num_bin(self) -> int:
        return len(self.bin_cols)

    def cat_vocab_sizes(self) -> List[int]:
        """Return vocab size (including missing id=0) for each cat column."""
        sizes = []
        for col in self.cat_cols:
            sizes.append(max(self.cat_mappings[col].values()) + 1)
        return sizes

    def bin_vocab_sizes(self) -> List[int]:
        """Return vocab size (including missing id=0) for each bin column."""
        sizes = []
        for col in self.bin_cols:
            sizes.append(max(self.bin_mappings[col].values()) + 1)
        return sizes


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PortoDataset(Dataset):
    def __init__(self, tensors: Dict[str, torch.Tensor]) -> None:
        self.tensors = tensors
        self._len = tensors["x_num"].shape[0]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tensors.items()}


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------

def load_porto_data(
    data_dir: str | Path,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[PortoPreprocessor, PortoDataset, PortoDataset]:
    """Load Porto Seguro train.csv, fit preprocessor, return train/valid datasets.

    Parameters
    ----------
    data_dir:
        Directory containing ``train.csv``.
    valid_size:
        Fraction of training rows to hold out for validation.
    random_state:
        Seed for stratified split.

    Returns
    -------
    preprocessor, train_dataset, valid_dataset
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    df = pd.read_csv(train_path)

    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=random_state,
        stratify=df["target"],
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    prep = PortoPreprocessor()
    train_tensors = prep.fit_transform(train_df)
    valid_tensors = prep.transform(valid_df)

    return prep, PortoDataset(train_tensors), PortoDataset(valid_tensors)
