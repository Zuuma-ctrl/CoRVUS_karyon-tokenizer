"""
Training and evaluation utilities.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
) -> float:
    """Run one full training epoch.

    Parameters
    ----------
    model    : MixedTabularModel (or compatible)
    loader   : DataLoader yielding dicts with keys x_num, missing_mask, x_cat, x_bin, y
    optimizer: any torch optimizer
    device   : torch device
    loss_fn  : loss function; defaults to BCEWithLogitsLoss

    Returns
    -------
    average training loss over the epoch
    """
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = _batch_to_device(batch, device)
        optimizer.zero_grad()
        logits, _ = model(
            batch["x_num"],
            batch["missing_mask"],
            batch["x_cat"],
            batch["x_bin"],
        )
        loss = loss_fn(logits, batch["y"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run inference and return predicted probabilities as a numpy array.

    Parameters
    ----------
    model  : MixedTabularModel (or compatible)
    loader : DataLoader (may or may not include 'y')
    device : torch device

    Returns
    -------
    preds : (N,) numpy float32 array of predicted probabilities
    """
    model.eval()
    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            logits, _ = model(
                batch["x_num"],
                batch["missing_mask"],
                batch["x_cat"],
                batch["x_bin"],
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)

    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# Get aux outputs (for analysis / visualisation)
# ---------------------------------------------------------------------------

def get_aux(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, List[np.ndarray]]:
    """Collect auxiliary outputs from the model over the full loader.

    Returns a dict where each value is a list of per-batch numpy arrays.
    To get the full array: ``np.concatenate(aux_dict['assign'], axis=0)``.
    """
    model.eval()
    aux_accum: Dict[str, List[np.ndarray]] = {}

    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            _, aux = model(
                batch["x_num"],
                batch["missing_mask"],
                batch["x_cat"],
                batch["x_bin"],
            )
            for k, v in aux.items():
                if isinstance(v, torch.Tensor):
                    arr = v.cpu().numpy()
                    aux_accum.setdefault(k, []).append(arr)

    return aux_accum
