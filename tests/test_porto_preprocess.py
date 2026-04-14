"""
Tests for Porto Seguro data preprocessing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.porto import (
    PortoPreprocessor,
    build_categorical_mappings,
    compute_numeric_stats,
    infer_schema,
)


# ---------------------------------------------------------------------------
# Helpers: build synthetic DataFrames that mimic Porto Seguro schema
# ---------------------------------------------------------------------------

def _make_synthetic_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "id": range(n),
            "target": rng.randint(0, 2, size=n),
            # numeric
            "ps_reg_01": rng.randn(n).astype(np.float32),
            "ps_reg_02": rng.randn(n).astype(np.float32),
            # categorical
            "ps_car_01_cat": rng.choice([-1, 1, 2, 3], size=n),
            "ps_car_02_cat": rng.choice([-1, 0, 1], size=n),
            # binary
            "ps_ind_06_bin": rng.choice([-1, 0, 1], size=n),
        }
    )
    # Inject some -1 (missing) values into numeric columns
    df.loc[rng.choice(n, 5, replace=False), "ps_reg_01"] = -1
    return df


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

class TestInferSchema:
    def test_target_column(self):
        schema = infer_schema(["target"])
        assert schema["target"] == "target"

    def test_id_column(self):
        schema = infer_schema(["id"])
        assert schema["id"] == "id"

    def test_cat_suffix(self):
        schema = infer_schema(["ps_car_01_cat"])
        assert schema["ps_car_01_cat"] == "categorical"

    def test_bin_suffix(self):
        schema = infer_schema(["ps_ind_06_bin"])
        assert schema["ps_ind_06_bin"] == "binary"

    def test_numeric_default(self):
        schema = infer_schema(["ps_reg_01", "ps_reg_02"])
        assert schema["ps_reg_01"] == "numeric"
        assert schema["ps_reg_02"] == "numeric"

    def test_full_synthetic_df(self):
        df = _make_synthetic_df()
        schema = infer_schema(list(df.columns))
        assert schema["target"] == "target"
        assert schema["id"] == "id"
        assert schema["ps_car_01_cat"] == "categorical"
        assert schema["ps_ind_06_bin"] == "binary"
        assert schema["ps_reg_01"] == "numeric"


# ---------------------------------------------------------------------------
# Numeric stats
# ---------------------------------------------------------------------------

class TestComputeNumericStats:
    def test_excludes_missing(self):
        df = pd.DataFrame({"x": [-1.0, 1.0, 2.0, 3.0, -1.0]})
        stats = compute_numeric_stats(df, ["x"], missing_value=-1.0)
        expected_mean = (1.0 + 2.0 + 3.0) / 3.0
        assert abs(stats["x"]["mean"] - expected_mean) < 1e-5

    def test_std_positive(self):
        df = _make_synthetic_df()
        stats = compute_numeric_stats(df, ["ps_reg_01", "ps_reg_02"])
        for col in ["ps_reg_01", "ps_reg_02"]:
            assert stats[col]["std"] > 0

    def test_all_missing_returns_defaults(self):
        df = pd.DataFrame({"x": [-1.0, -1.0, -1.0]})
        stats = compute_numeric_stats(df, ["x"])
        # Should not crash; mean=0, std=1 as fallback
        assert stats["x"]["std"] > 0


# ---------------------------------------------------------------------------
# Categorical mappings
# ---------------------------------------------------------------------------

class TestBuildCategoricalMappings:
    def test_missing_mapped_to_zero(self):
        df = pd.DataFrame({"cat": [-1, 0, 1, 2, -1]})
        mappings = build_categorical_mappings(df, ["cat"])
        assert mappings["cat"][-1] == 0

    def test_other_values_positive(self):
        df = pd.DataFrame({"cat": [-1, 0, 1, 2]})
        mappings = build_categorical_mappings(df, ["cat"])
        for k, v in mappings["cat"].items():
            if k != -1:
                assert v >= 1


# ---------------------------------------------------------------------------
# PortoPreprocessor
# ---------------------------------------------------------------------------

class TestPortoPreprocessor:
    def test_fit_transform_shape(self):
        df = _make_synthetic_df(n=100)
        prep = PortoPreprocessor()
        result = prep.fit_transform(df)

        N = 100
        assert result["x_num"].shape == (N, prep.num_numeric)
        assert result["missing_mask"].shape == (N, prep.num_numeric)
        assert result["x_cat"].shape == (N, prep.num_cat)
        assert result["x_bin"].shape == (N, prep.num_bin)
        assert result["y"].shape == (N,)

    def test_missing_mask_correct(self):
        """Rows where the original value was -1 should have True in missing_mask."""
        df = _make_synthetic_df(n=100)
        prep = PortoPreprocessor()
        result = prep.fit_transform(df)

        col_idx = prep.numeric_cols.index("ps_reg_01")
        original = df["ps_reg_01"].to_numpy()
        expected_missing = original == -1.0
        actual_missing = result["missing_mask"][:, col_idx].numpy()
        np.testing.assert_array_equal(actual_missing, expected_missing)

    def test_x_num_dtype(self):
        df = _make_synthetic_df()
        prep = PortoPreprocessor()
        result = prep.fit_transform(df)
        assert result["x_num"].dtype == torch.float32

    def test_x_cat_dtype(self):
        df = _make_synthetic_df()
        prep = PortoPreprocessor()
        result = prep.fit_transform(df)
        assert result["x_cat"].dtype == torch.long

    def test_no_id_in_numeric(self):
        df = _make_synthetic_df()
        prep = PortoPreprocessor()
        prep.fit(df)
        assert "id" not in prep.numeric_cols

    def test_no_target_in_numeric(self):
        df = _make_synthetic_df()
        prep = PortoPreprocessor()
        prep.fit(df)
        assert "target" not in prep.numeric_cols

    def test_transform_without_target(self):
        """transform should work even when 'target' column is absent."""
        df = _make_synthetic_df(n=20)
        prep = PortoPreprocessor()
        prep.fit(df)
        df_no_target = df.drop(columns=["target"])
        result = prep.transform(df_no_target)
        assert "y" not in result

    def test_cat_vocab_sizes_include_missing(self):
        df = _make_synthetic_df()
        prep = PortoPreprocessor()
        prep.fit(df)
        for size in prep.cat_vocab_sizes():
            assert size >= 1  # at least the missing id

    def test_no_nan_in_x_num(self):
        df = _make_synthetic_df()
        prep = PortoPreprocessor()
        result = prep.fit_transform(df)
        assert not torch.isnan(result["x_num"]).any()
