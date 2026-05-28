from __future__ import annotations

import torch
from torch import nn


class SATBGRULanguageModel(nn.Module):
    """Autoregressive GRU for unconditioned SATB generation."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=embedding_dim * 4,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, vocab_size) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = [self.embedding(x[:, :, voice]) for voice in range(4)]
        combined = torch.cat(embeddings, dim=-1)
        hidden, _ = self.gru(combined)
        return tuple(head(hidden) for head in self.heads)


class SopranoConditionedHarmonizer(nn.Module):
    """Bidirectional GRU that predicts alto, tenor, and bass from soprano."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        factor = 2 if bidirectional else 1
        self.heads = nn.ModuleList([nn.Linear(hidden_dim * factor, vocab_size) for _ in range(3)])

    def forward(self, soprano: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(soprano)
        hidden, _ = self.gru(embedded)
        return tuple(head(hidden) for head in self.heads)

