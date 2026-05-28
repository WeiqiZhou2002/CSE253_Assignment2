from __future__ import annotations

import torch


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    banned_ids: list[int] | None = None,
) -> torch.Tensor:
    """Temperature sample one token id from a 1D logits tensor."""
    if temperature <= 0:
        temperature = 1.0
    adjusted = logits.float().clone() / temperature
    for token_id in banned_ids or []:
        adjusted[int(token_id)] = -1e9
    probs = torch.softmax(adjusted, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(0)


def argmax_from_logits(logits: torch.Tensor, banned_ids: list[int] | None = None) -> torch.Tensor:
    """Choose the highest-scoring token, optionally masking special ids."""
    adjusted = logits.float().clone()
    for token_id in banned_ids or []:
        adjusted[int(token_id)] = -1e9
    return torch.argmax(adjusted, dim=-1)

