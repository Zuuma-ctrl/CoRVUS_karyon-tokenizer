"""
Tests for numeric tokenizers:
- OrderedThresholdTokenizer thresholds are monotonically increasing
- Output shapes are correct for all tokenizers
- Missing mask is correctly handled
- aux dict contains expected keys
"""

from __future__ import annotations

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.numeric_tokenizers import (
    CenterSoftBinTokenizer,
    LinearNumericTokenizer,
    OrderedThresholdTokenizer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 8
NUM_FEATURES = 5
D_TOKEN = 16
NUM_CENTERS = 4
NUM_THRESHOLDS = 6


@pytest.fixture
def sample_inputs():
    torch.manual_seed(0)
    x_num = torch.randn(BATCH, NUM_FEATURES)
    # Mark some entries as missing
    missing_mask = torch.zeros(BATCH, NUM_FEATURES, dtype=torch.bool)
    missing_mask[0, 0] = True
    missing_mask[3, 2] = True
    return x_num, missing_mask


# ---------------------------------------------------------------------------
# LinearNumericTokenizer
# ---------------------------------------------------------------------------

class TestLinearNumericTokenizer:
    def test_output_shape(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = LinearNumericTokenizer(NUM_FEATURES, D_TOKEN)
        tokens, aux = tok(x_num, missing_mask)
        assert tokens.shape == (BATCH, NUM_FEATURES, D_TOKEN), (
            f"Expected {(BATCH, NUM_FEATURES, D_TOKEN)}, got {tokens.shape}"
        )

    def test_aux_is_empty_dict(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = LinearNumericTokenizer(NUM_FEATURES, D_TOKEN)
        _, aux = tok(x_num, missing_mask)
        assert isinstance(aux, dict)
        assert len(aux) == 0

    def test_missing_mask_applied(self, sample_inputs):
        """Missing positions should produce the same embedding regardless of x value."""
        x_num, missing_mask = sample_inputs
        tok = LinearNumericTokenizer(NUM_FEATURES, D_TOKEN)
        tok.eval()

        # Two different x_num values at a missing position
        x_alt = x_num.clone()
        x_alt[0, 0] = x_num[0, 0] + 100.0  # very different value at missing spot

        with torch.no_grad():
            tokens1, _ = tok(x_num, missing_mask)
            tokens2, _ = tok(x_alt, missing_mask)

        # Tokens at missing position should be identical
        assert torch.allclose(tokens1[0, 0], tokens2[0, 0]), (
            "Missing position should produce the same token regardless of input value."
        )

    def test_no_nan_in_output(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = LinearNumericTokenizer(NUM_FEATURES, D_TOKEN)
        tokens, _ = tok(x_num, missing_mask)
        assert not torch.isnan(tokens).any()


# ---------------------------------------------------------------------------
# CenterSoftBinTokenizer
# ---------------------------------------------------------------------------

class TestCenterSoftBinTokenizer:
    def test_output_shape(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = CenterSoftBinTokenizer(NUM_FEATURES, D_TOKEN, NUM_CENTERS)
        tokens, aux = tok(x_num, missing_mask)
        assert tokens.shape == (BATCH, NUM_FEATURES, D_TOKEN)

    def test_aux_is_empty_dict(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = CenterSoftBinTokenizer(NUM_FEATURES, D_TOKEN, NUM_CENTERS)
        _, aux = tok(x_num, missing_mask)
        assert isinstance(aux, dict)
        assert len(aux) == 0

    def test_missing_mask_applied(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = CenterSoftBinTokenizer(NUM_FEATURES, D_TOKEN, NUM_CENTERS)
        tok.eval()

        x_alt = x_num.clone()
        x_alt[0, 0] = x_num[0, 0] + 1000.0

        with torch.no_grad():
            tokens1, _ = tok(x_num, missing_mask)
            tokens2, _ = tok(x_alt, missing_mask)

        assert torch.allclose(tokens1[0, 0], tokens2[0, 0])

    def test_no_nan_in_output(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = CenterSoftBinTokenizer(NUM_FEATURES, D_TOKEN, NUM_CENTERS)
        tokens, _ = tok(x_num, missing_mask)
        assert not torch.isnan(tokens).any()


# ---------------------------------------------------------------------------
# OrderedThresholdTokenizer
# ---------------------------------------------------------------------------

class TestOrderedThresholdTokenizer:
    def test_output_shape(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = OrderedThresholdTokenizer(NUM_FEATURES, D_TOKEN, NUM_THRESHOLDS)
        tokens, aux = tok(x_num, missing_mask)
        assert tokens.shape == (BATCH, NUM_FEATURES, D_TOKEN)

    def test_thresholds_monotonically_increasing(self):
        """Thresholds must be strictly increasing for every feature."""
        tok = OrderedThresholdTokenizer(NUM_FEATURES, D_TOKEN, NUM_THRESHOLDS)
        thresholds = tok.get_thresholds()  # (M, K)
        assert thresholds.shape == (NUM_FEATURES, NUM_THRESHOLDS)
        # diff along K dimension should be positive
        diffs = thresholds[:, 1:] - thresholds[:, :-1]  # (M, K-1)
        assert (diffs > 0).all(), "Thresholds are not strictly increasing."

    def test_thresholds_stay_ordered_after_training_step(self, sample_inputs):
        """Thresholds should remain ordered even after a gradient update."""
        x_num, missing_mask = sample_inputs
        tok = OrderedThresholdTokenizer(NUM_FEATURES, D_TOKEN, NUM_THRESHOLDS)
        optimizer = torch.optim.Adam(tok.parameters(), lr=1e-2)

        for _ in range(5):
            tokens, _ = tok(x_num, missing_mask)
            loss = tokens.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        thresholds = tok.get_thresholds()
        diffs = thresholds[:, 1:] - thresholds[:, :-1]
        assert (diffs > 0).all(), "Thresholds became non-monotone after training."

    def test_aux_keys(self, sample_inputs):
        """aux must contain 'assign' and 'thresholds'."""
        x_num, missing_mask = sample_inputs
        tok = OrderedThresholdTokenizer(NUM_FEATURES, D_TOKEN, NUM_THRESHOLDS)
        _, aux = tok(x_num, missing_mask)
        assert "assign" in aux, "aux missing key 'assign'"
        assert "thresholds" in aux, "aux missing key 'thresholds'"

    def test_aux_assign_shape(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = OrderedThresholdTokenizer(NUM_FEATURES, D_TOKEN, NUM_THRESHOLDS)
        _, aux = tok(x_num, missing_mask)
        assert aux["assign"].shape == (BATCH, NUM_FEATURES, NUM_THRESHOLDS)

    def test_aux_thresholds_shape(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = OrderedThresholdTokenizer(NUM_FEATURES, D_TOKEN, NUM_THRESHOLDS)
        _, aux = tok(x_num, missing_mask)
        assert aux["thresholds"].shape == (NUM_FEATURES, NUM_THRESHOLDS)

    def test_missing_mask_applied(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = OrderedThresholdTokenizer(NUM_FEATURES, D_TOKEN, NUM_THRESHOLDS)
        tok.eval()

        x_alt = x_num.clone()
        x_alt[0, 0] = x_num[0, 0] + 1000.0

        with torch.no_grad():
            tokens1, _ = tok(x_num, missing_mask)
            tokens2, _ = tok(x_alt, missing_mask)

        assert torch.allclose(tokens1[0, 0], tokens2[0, 0])

    def test_no_nan_in_output(self, sample_inputs):
        x_num, missing_mask = sample_inputs
        tok = OrderedThresholdTokenizer(NUM_FEATURES, D_TOKEN, NUM_THRESHOLDS)
        tokens, _ = tok(x_num, missing_mask)
        assert not torch.isnan(tokens).any()

    def test_assign_values_in_0_1(self, sample_inputs):
        """Soft assignment values should be in [0, 1] (sigmoid output)."""
        x_num, missing_mask = sample_inputs
        tok = OrderedThresholdTokenizer(NUM_FEATURES, D_TOKEN, NUM_THRESHOLDS)
        _, aux = tok(x_num, missing_mask)
        assign = aux["assign"]
        assert (assign >= 0).all() and (assign <= 1).all()
