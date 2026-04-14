"""
Numeric tokenizers for tabular data.

All tokenizers share the same interface::

    tokens, aux = tokenizer(x_num, missing_mask)

    tokens       : (batch, num_features, d_token)
    aux          : dict  - keys depend on tokenizer type
                   OrderedThresholdTokenizer returns 'assign' and 'thresholds'

Three implementations:
  LinearNumericTokenizer  - simplest linear projection baseline
  CenterSoftBinTokenizer  - soft membership to learned center points
  OrderedThresholdTokenizer - ordered thresholds (closest to decision-tree splits)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Linear
# ---------------------------------------------------------------------------

class LinearNumericTokenizer(nn.Module):
    """Per-feature linear projection: x -> W * x + b.

    Missing values are replaced by a dedicated learned embedding.
    """

    def __init__(self, num_features: int, d_token: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        # One weight and bias per feature (not shared across features)
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))
        self.missing_emb = nn.Parameter(torch.empty(num_features, d_token))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.missing_emb, std=0.02)

    def forward(
        self,
        x_num: torch.Tensor,       # (B, M)
        missing_mask: torch.Tensor, # (B, M) bool - True where missing
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns
        -------
        tokens : (B, M, d_token)
        aux    : {}
        """
        B, M = x_num.shape
        # (B, M, 1) * (M, d) -> (B, M, d)
        tokens = x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        # Replace missing positions with dedicated embedding
        # missing_emb: (M, d) -> (1, M, d)
        miss = self.missing_emb.unsqueeze(0).expand(B, -1, -1)
        mask = missing_mask.unsqueeze(-1).expand_as(tokens)
        tokens = torch.where(mask, miss, tokens)
        return tokens, {}


# ---------------------------------------------------------------------------
# 2. Center-based Soft Bins
# ---------------------------------------------------------------------------

class CenterSoftBinTokenizer(nn.Module):
    """Soft membership to K learned center points per feature.

    For each feature m and each center k:
        similarity[m, k] = exp(-distance(x_m, center[m,k])^2)
    Membership is normalised across centers (softmax).
    The output token for feature m is: sum_k membership[m,k] * emb[m,k].

    Missing values are replaced by a dedicated learned embedding.
    """

    def __init__(self, num_features: int, d_token: int, num_centers: int = 8) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        self.num_centers = num_centers

        # centers: (M, K)
        self.centers = nn.Parameter(torch.randn(num_features, num_centers))
        # per-center embeddings: (M, K, d)
        self.center_emb = nn.Parameter(torch.empty(num_features, num_centers, d_token))
        self.missing_emb = nn.Parameter(torch.empty(num_features, d_token))
        # log-scale bandwidth (learnable)
        self.log_bandwidth = nn.Parameter(torch.zeros(num_features))

        nn.init.normal_(self.center_emb, std=0.02)
        nn.init.normal_(self.missing_emb, std=0.02)

    def forward(
        self,
        x_num: torch.Tensor,       # (B, M)
        missing_mask: torch.Tensor, # (B, M) bool
    ) -> Tuple[torch.Tensor, Dict]:
        B, M = x_num.shape
        K = self.num_centers

        # bandwidth: (M,) positive
        bandwidth = self.log_bandwidth.exp() + 1e-6  # (M,)

        # x: (B, M, 1) - centers: (1, M, K) -> dist2: (B, M, K)
        x_expanded = x_num.unsqueeze(-1)             # (B, M, 1)
        centers = self.centers.unsqueeze(0)          # (1, M, K)
        dist2 = ((x_expanded - centers) ** 2) / bandwidth.unsqueeze(0).unsqueeze(-1)

        # soft membership: (B, M, K)
        membership = torch.softmax(-dist2, dim=-1)

        # tokens: (B, M, d)
        # center_emb: (1, M, K, d)  membership: (B, M, K, 1)
        center_emb = self.center_emb.unsqueeze(0)    # (1, M, K, d)
        tokens = (membership.unsqueeze(-1) * center_emb).sum(dim=2)  # (B, M, d)

        # Replace missing positions
        miss = self.missing_emb.unsqueeze(0).expand(B, -1, -1)
        mask = missing_mask.unsqueeze(-1).expand_as(tokens)
        tokens = torch.where(mask, miss, tokens)
        return tokens, {}


# ---------------------------------------------------------------------------
# 3. Ordered Threshold Tokenizer (main proposal)
# ---------------------------------------------------------------------------

class OrderedThresholdTokenizer(nn.Module):
    """Ordered-thresholds numeric tokenizer.

    For each feature m we maintain K ordered thresholds::

        base[m]            learnable scalar
        delta_raw[m, k]    learnable (raw, unconstrained)
        delta = softplus(delta_raw) + eps   (always positive)
        thresholds[m, :] = base[m] + cumsum(delta[m, :])

    The soft interval assignment for value x_m is::

        h[m, k] = sigmoid((x_m - threshold[m, k]) / temperature)

    This produces a monotone vector in (0,1)^K.  The output token is then a
    linear projection of h (via per-feature weight matrix W_proj).

    Missing values are replaced by a dedicated learned embedding.

    Auxiliary outputs (``aux`` dict)::
        'assign'     : (B, M, K) soft interval assignment (observation only)
        'thresholds' : (M, K) learned thresholds (detached)
    """

    def __init__(
        self,
        num_features: int,
        d_token: int,
        num_thresholds: int = 8,
        temperature: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        self.num_thresholds = num_thresholds
        self.temperature = temperature
        self.eps = eps

        # threshold parameters
        self.base = nn.Parameter(torch.zeros(num_features))
        self.delta_raw = nn.Parameter(torch.zeros(num_features, num_thresholds))

        # projection: (M, K, d)
        self.proj = nn.Parameter(torch.empty(num_features, num_thresholds, d_token))
        self.proj_bias = nn.Parameter(torch.zeros(num_features, d_token))

        self.missing_emb = nn.Parameter(torch.empty(num_features, d_token))

        nn.init.normal_(self.proj, std=0.02)
        nn.init.normal_(self.missing_emb, std=0.02)

    def get_thresholds(self) -> torch.Tensor:
        """Return ordered thresholds of shape (M, K)."""
        delta = F.softplus(self.delta_raw) + self.eps          # (M, K) positive
        thresholds = self.base.unsqueeze(-1) + torch.cumsum(delta, dim=-1)  # (M, K)
        return thresholds

    def forward(
        self,
        x_num: torch.Tensor,       # (B, M)
        missing_mask: torch.Tensor, # (B, M) bool
    ) -> Tuple[torch.Tensor, Dict]:
        B, M = x_num.shape
        K = self.num_thresholds

        thresholds = self.get_thresholds()  # (M, K)

        # soft interval assignment: (B, M, K)
        # x_num: (B, M, 1)  thresholds: (1, M, K)
        x_exp = x_num.unsqueeze(-1)                              # (B, M, 1)
        thresh_exp = thresholds.unsqueeze(0)                     # (1, M, K)
        assign = torch.sigmoid((x_exp - thresh_exp) / self.temperature)  # (B, M, K)

        # project to d_token: assign (B,M,K) x proj (M,K,d) -> (B,M,d)
        # use einsum: b m k, m k d -> b m d
        tokens = torch.einsum("bmk,mkd->bmd", assign, self.proj) + self.proj_bias.unsqueeze(0)

        # Replace missing positions
        miss = self.missing_emb.unsqueeze(0).expand(B, -1, -1)
        mask = missing_mask.unsqueeze(-1).expand_as(tokens)
        tokens = torch.where(mask, miss, tokens)

        aux = {
            "assign": assign,                           # (B, M, K) - raw values including missing positions (not masked)
            "thresholds": thresholds.detach(),          # (M, K)
        }
        return tokens, aux
