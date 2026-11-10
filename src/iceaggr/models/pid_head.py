"""PID classification head: CLS summary -> one score (logit) per class."""

import torch
import torch.nn as nn


class PIDHead(nn.Module):
    """Two-layer MLP that maps the event embedding to class scores (logits).

    Output is NOT normalized: nn.functional.cross_entropy expects raw logits
    and applies the softmax itself.
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, event_embedding: torch.Tensor) -> torch.Tensor:
        """(batch, embed_dim) -> (batch, num_classes) logits."""
        x = self.fc1(event_embedding)
        x = self.activation(x)
        return self.fc2(x)