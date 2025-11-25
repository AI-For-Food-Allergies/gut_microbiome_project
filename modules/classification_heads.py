"""
Classification heads for microbiome models.

These heads are small PyTorch modules that take a sample-level
feature vector and return logits for classification.
"""

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """
    Simple linear classifier (logistic-style).

    Input:
        features: (B, input_dim)
    Output:
        logits:   (B, num_classes)
    """

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class MLPHead(nn.Module):
    """
    Two-layer MLP classifier with ReLU and dropout.

    Input:
        features: (B, input_dim)
    Output:
        logits:   (B, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
