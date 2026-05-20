from __future__ import annotations


class MissingTorchError(ImportError):
    pass


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise MissingTorchError("Install the deep extra: pip install -e '.[deep]'") from exc
    return torch, nn


def build_sequence_expression_regressor(
    *,
    context_dim: int,
    sequence_channels: int = 4,
    hidden_dim: int = 256,
    dropout: float = 0.1,
):
    """Build a compact sequence + expression PyTorch model.

    Input tensors:
    - sequence_onehot: float tensor shaped [batch, 4, length]
    - context_vector: float tensor shaped [batch, context_dim]
    """
    torch, nn = require_torch()

    class SequenceExpressionRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.sequence_encoder = nn.Sequential(
                nn.Conv1d(sequence_channels, 128, kernel_size=9, padding=4),
                nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=9, padding=8, dilation=2),
                nn.GELU(),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
                nn.Linear(128, hidden_dim),
                nn.GELU(),
            )
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, sequence_onehot, context_vector):
            seq_latent = self.sequence_encoder(sequence_onehot)
            ctx_latent = self.context_encoder(context_vector)
            gated_seq = seq_latent * self.gate(ctx_latent)
            return self.head(torch.cat([gated_seq, ctx_latent], dim=1)).squeeze(-1)

    return SequenceExpressionRegressor()
