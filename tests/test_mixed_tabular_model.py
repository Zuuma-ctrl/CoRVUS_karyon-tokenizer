"""
Tests for MixedTabularModel:
- All modes (linear, center, ordered) run forward pass without errors
- Output shapes are correct
- aux is returned
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.mixed_tabular_model import MixedTabularModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 6
NUM_NUMERIC = 4
D_TOKEN = 16
CAT_VOCAB_SIZES = [5, 3]   # 2 categorical features
BIN_VOCAB_SIZES = [3, 2]   # 2 binary features


@pytest.fixture
def sample_batch():
    torch.manual_seed(1)
    x_num = torch.randn(BATCH, NUM_NUMERIC)
    missing_mask = torch.zeros(BATCH, NUM_NUMERIC, dtype=torch.bool)
    missing_mask[0, 1] = True
    x_cat = torch.randint(0, 3, (BATCH, len(CAT_VOCAB_SIZES)))
    x_bin = torch.randint(0, 2, (BATCH, len(BIN_VOCAB_SIZES)))
    return x_num, missing_mask, x_cat, x_bin


def _make_model(mode: str) -> MixedTabularModel:
    return MixedTabularModel(
        num_numeric=NUM_NUMERIC,
        cat_vocab_sizes=CAT_VOCAB_SIZES,
        bin_vocab_sizes=BIN_VOCAB_SIZES,
        d_token=D_TOKEN,
        mlp_hidden=[64, 32],
        mode=mode,
        num_centers=4,
        num_thresholds=4,
        dropout=0.0,  # deterministic for testing
    )


# ---------------------------------------------------------------------------
# Parametrised tests over all modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["linear", "center", "ordered"])
class TestMixedTabularModelForward:
    def test_logits_shape(self, mode, sample_batch):
        x_num, missing_mask, x_cat, x_bin = sample_batch
        model = _make_model(mode)
        model.eval()
        with torch.no_grad():
            logits, aux = model(x_num, missing_mask, x_cat, x_bin)
        assert logits.shape == (BATCH,), f"[{mode}] Expected ({BATCH},), got {logits.shape}"

    def test_aux_is_dict(self, mode, sample_batch):
        x_num, missing_mask, x_cat, x_bin = sample_batch
        model = _make_model(mode)
        model.eval()
        with torch.no_grad():
            _, aux = model(x_num, missing_mask, x_cat, x_bin)
        assert isinstance(aux, dict), f"[{mode}] aux should be a dict"

    def test_no_nan_in_logits(self, mode, sample_batch):
        x_num, missing_mask, x_cat, x_bin = sample_batch
        model = _make_model(mode)
        model.eval()
        with torch.no_grad():
            logits, _ = model(x_num, missing_mask, x_cat, x_bin)
        assert not torch.isnan(logits).any(), f"[{mode}] NaN in logits"

    def test_backward_runs(self, mode, sample_batch):
        """Gradient should flow without errors."""
        x_num, missing_mask, x_cat, x_bin = sample_batch
        model = _make_model(mode)
        logits, _ = model(x_num, missing_mask, x_cat, x_bin)
        target = torch.zeros(BATCH)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()  # should not raise


class TestOrderedModeAux:
    def test_ordered_aux_keys(self, sample_batch):
        x_num, missing_mask, x_cat, x_bin = sample_batch
        model = _make_model("ordered")
        model.eval()
        with torch.no_grad():
            _, aux = model(x_num, missing_mask, x_cat, x_bin)
        assert "assign" in aux, "ordered model aux missing 'assign'"
        assert "thresholds" in aux, "ordered model aux missing 'thresholds'"

    def test_ordered_assign_shape(self, sample_batch):
        x_num, missing_mask, x_cat, x_bin = sample_batch
        model = _make_model("ordered")
        model.eval()
        with torch.no_grad():
            _, aux = model(x_num, missing_mask, x_cat, x_bin)
        assert aux["assign"].shape == (BATCH, NUM_NUMERIC, 4)  # num_thresholds=4

    def test_ordered_thresholds_monotone(self, sample_batch):
        x_num, missing_mask, x_cat, x_bin = sample_batch
        model = _make_model("ordered")
        thresholds = model.numeric_tokenizer.get_thresholds()  # (M, K)
        diffs = thresholds[:, 1:] - thresholds[:, :-1]
        assert (diffs > 0).all(), "Thresholds not monotone in model"


class TestLinearAndCenterAux:
    @pytest.mark.parametrize("mode", ["linear", "center"])
    def test_aux_empty(self, mode, sample_batch):
        x_num, missing_mask, x_cat, x_bin = sample_batch
        model = _make_model(mode)
        model.eval()
        with torch.no_grad():
            _, aux = model(x_num, missing_mask, x_cat, x_bin)
        assert len(aux) == 0, f"[{mode}] Expected empty aux, got keys: {list(aux.keys())}"


class TestInvalidMode:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            MixedTabularModel(
                num_numeric=NUM_NUMERIC,
                cat_vocab_sizes=CAT_VOCAB_SIZES,
                bin_vocab_sizes=BIN_VOCAB_SIZES,
                mode="invalid_mode",
            )


class TestNoNumericFeatures:
    def test_forward_no_numeric(self):
        """Model should work when there are no numeric features."""
        model = MixedTabularModel(
            num_numeric=0,
            cat_vocab_sizes=[3, 5],
            bin_vocab_sizes=[2],
            d_token=D_TOKEN,
            mlp_hidden=[32],
            mode="ordered",
        )
        model.eval()
        x_num = torch.zeros(BATCH, 0)
        missing_mask = torch.zeros(BATCH, 0, dtype=torch.bool)
        x_cat = torch.randint(0, 2, (BATCH, 2))
        x_bin = torch.randint(0, 2, (BATCH, 1))
        with torch.no_grad():
            logits, aux = model(x_num, missing_mask, x_cat, x_bin)
        assert logits.shape == (BATCH,)
