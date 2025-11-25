"""
High-level classifier integrating:
- MicrobiomeTransformer (per-OTU outputs)
- Feature aggregation
- Classification head
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .model import MicrobiomeTransformer
from .classification_heads import LinearHead, MLPHead


class MicrobiomeClassifier(nn.Module):
    """
    Wraps:
        transformer → per-OTU scores → feature aggregation → classifier head

    Expected batch keys (same as MicrobiomeTransformer):
        - 'embeddings_type1': (B, L1, D1)
        - 'embeddings_type2': (B, L2, D2)
        - 'mask':             (B, L1 + L2)  (True = valid, False = padding)
        - 'type_indicators':  (B, L1 + L2)  (not used here yet, but available)

    Forward:
        logits = MicrobiomeClassifier(batch)  # (B, num_classes)
    """

    def __init__(
        self,
        model: MicrobiomeTransformer,
        classification_head_type: str = "linear",
        num_classes: int = 2,
    ):
        super().__init__()
        self.model = model
        self.classification_head_type = classification_head_type
        self.num_classes = num_classes

        # We aggregate into a 3D feature: [mean, max, std] of per-OTU scores
        self.feature_dim = 3

        self.classification_head = self.init_classification_head()

    # -----------------------------
    # Classification head selection
    # -----------------------------
    def init_classification_head(self) -> nn.Module:
        """
        Initialize the classifier head based on the chosen type.
        """
        if self.classification_head_type == "linear":
            return LinearHead(self.feature_dim, num_classes=self.num_classes)
        elif self.classification_head_type == "mlp":
            return MLPHead(self.feature_dim, hidden_dim=64, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown classification_head_type: {self.classification_head_type}")

    # -----------------------------
    # Feature aggregation
    # -----------------------------
    @staticmethod
    def aggregate_scores(
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate per-OTU scores into sample-level features.

        Args:
            scores: (B, L) - output from MicrobiomeTransformer
            mask:   (B, L) - True for valid positions, False for padding

        Returns:
            features: (B, 3) with [mean, max, std] per sample
        """
        # Ensure float and avoid division by zero
        mask_f = mask.float()  # (B, L)
        denom = mask_f.sum(dim=1).clamp_min(1.0)  # (B,)

        # Mean
        summed = (scores * mask_f).sum(dim=1)  # (B,)
        mean = summed / denom  # (B,)

        # Max: set padded positions to very small number before max
        # Expand mask to use -inf where mask is False
        neg_inf = torch.finfo(scores.dtype).min
        masked_for_max = torch.where(mask, scores, torch.full_like(scores, neg_inf))
        max_vals, _ = masked_for_max.max(dim=1)  # (B,)

        # Std: sqrt(E[x^2] - mean^2) over active positions
        squared = (scores ** 2) * mask_f  # (B, L)
        mean_sq = squared.sum(dim=1) / denom  # (B,)
        var = (mean_sq - mean ** 2).clamp_min(0.0)
        std = torch.sqrt(var)  # (B,)

        features = torch.stack([mean, max_vals, std], dim=1)  # (B, 3)
        return features

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute logits for each sample in the batch.

        Returns:
            logits: (B, num_classes)
        """
        # 1) Transformer: per-OTU scores
        scores = self.model(batch)  # (B, L)

        # 2) Aggregate to sample-level features
        mask = batch["mask"]  # (B, L)
        features = self.aggregate_scores(scores, mask)  # (B, 3)

        # 3) Classifier head
        logits = self.classification_head(features)  # (B, num_classes)
        return logits
