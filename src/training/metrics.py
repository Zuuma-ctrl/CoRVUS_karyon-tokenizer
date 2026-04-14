"""
Evaluation metrics for binary classification and regression.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    log_loss,
    r2_score,
    roc_auc_score,
)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_score))


def compute_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary cross-entropy log loss."""
    # clip for numerical stability
    y_score = np.clip(y_score, 1e-7, 1 - 1e-7)
    return float(log_loss(y_true, y_score))


def compute_rmse(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_score) ** 2)))


def compute_mae(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_score)))


def compute_r2(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """R² coefficient of determination."""
    return float(r2_score(y_true, y_score))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute all metrics and return as a dict.

    Parameters
    ----------
    y_true : binary labels (0/1)
    y_score : predicted probabilities in [0, 1]
    """
    return {
        "auc": compute_auc(y_true, y_score),
        "logloss": compute_logloss(y_true, y_score),
        "rmse": compute_rmse(y_true, y_score),
        "mae": compute_mae(y_true, y_score),
        "r2": compute_r2(y_true, y_score),
    }
