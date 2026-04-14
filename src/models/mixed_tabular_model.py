"""
MixedTabularModel - shared backbone for numeric/categorical/binary data.

Numeric branch: switchable tokenizer (linear | center | ordered)
Categorical branch: per-column embedding
Binary branch: per-column embedding
All tokens are concatenated and flattened, then passed through a small MLP head.

Forward returns (logits, aux) for binary classification (sigmoid output).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .numeric_tokenizers import (
    CenterSoftBinTokenizer,
    LinearNumericTokenizer,
    OrderedThresholdTokenizer,
)


class MixedTabularModel(nn.Module):
    """Mixed tabular model with interchangeable numeric tokenizer.

    Parameters
    ----------
    num_numeric : int
        Number of numeric features.
    cat_vocab_sizes : List[int]
        Vocabulary size (incl. missing id=0) for each categorical feature.
    bin_vocab_sizes : List[int]
        Vocabulary size (incl. missing id=0) for each binary feature.
    d_token : int
        Token dimension for all branches.
    mlp_hidden : List[int]
        Hidden layer sizes for the MLP head.
    mode : str
        Numeric tokenizer mode: 'linear' | 'center' | 'ordered'.
    num_centers : int
        Number of center points (for 'center' mode).
    num_thresholds : int
        Number of thresholds (for 'ordered' mode).
    dropout : float
        Dropout rate in MLP head.
    """

    def __init__(
        self,
        num_numeric: int,
        cat_vocab_sizes: List[int],
        bin_vocab_sizes: List[int],
        d_token: int = 32,
        mlp_hidden: Optional[List[int]] = None,
        mode: str = "ordered",
        num_centers: int = 8,
        num_thresholds: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_numeric = num_numeric
        self.cat_vocab_sizes = cat_vocab_sizes
        self.bin_vocab_sizes = bin_vocab_sizes
        self.d_token = d_token
        self.mode = mode

        if mlp_hidden is None:
            mlp_hidden = [256, 128]

        # ---- Numeric tokenizer ----
        if num_numeric > 0:
            if mode == "linear":
                self.numeric_tokenizer: nn.Module = LinearNumericTokenizer(num_numeric, d_token)
            elif mode == "center":
                self.numeric_tokenizer = CenterSoftBinTokenizer(num_numeric, d_token, num_centers)
            elif mode == "ordered":
                self.numeric_tokenizer = OrderedThresholdTokenizer(
                    num_numeric, d_token, num_thresholds
                )
            else:
                raise ValueError(f"Unknown tokenizer mode: {mode!r}. Choose linear|center|ordered.")
        else:
            self.numeric_tokenizer = None  # type: ignore[assignment]

        # ---- Categorical embeddings ----
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, d_token) for vocab_size in cat_vocab_sizes]
        )

        # ---- Binary embeddings ----
        self.bin_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, d_token) for vocab_size in bin_vocab_sizes]
        )

        # ---- MLP head ----
        num_tokens = num_numeric + len(cat_vocab_sizes) + len(bin_vocab_sizes)
        in_dim = num_tokens * d_token
        layers: List[nn.Module] = []
        prev = in_dim
        for h in mlp_hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_num: torch.Tensor,        # (B, num_numeric)
        missing_mask: torch.Tensor, # (B, num_numeric) bool
        x_cat: torch.Tensor,        # (B, num_cat)
        x_bin: torch.Tensor,        # (B, num_bin)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Returns
        -------
        logits : (B,)  raw sigmoid logit (binary classification)
        aux    : dict  forwarded from numeric tokenizer
        """
        tokens_list = []
        aux: Dict = {}

        # ---- Numeric tokens ----
        if self.num_numeric > 0 and self.numeric_tokenizer is not None:
            num_tokens, aux = self.numeric_tokenizer(x_num, missing_mask)  # (B, M, d)
            tokens_list.append(num_tokens)

        # ---- Categorical tokens ----
        for i, emb in enumerate(self.cat_embeddings):
            tokens_list.append(emb(x_cat[:, i]).unsqueeze(1))  # (B, 1, d)

        # ---- Binary tokens ----
        for i, emb in enumerate(self.bin_embeddings):
            tokens_list.append(emb(x_bin[:, i]).unsqueeze(1))  # (B, 1, d)

        # ---- Flatten and MLP ----
        if tokens_list:
            all_tokens = torch.cat(tokens_list, dim=1)  # (B, T, d)
            flat = all_tokens.flatten(1)                 # (B, T*d)
        else:
            # edge case: no features at all
            flat = x_num.new_zeros((x_num.shape[0], 1))

        logits = self.mlp(flat).squeeze(-1)  # (B,)
        return logits, aux
