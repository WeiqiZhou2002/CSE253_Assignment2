from __future__ import annotations

import torch


def sample_top_k_temperature(logits: torch.Tensor, temperature: float = 0.7, top_k: int = 5) -> int:
    """Sample one token from 1D logits using temperature scaling and top-k filtering."""
    if logits.ndim != 1:
        raise ValueError("sample_top_k_temperature expects a 1D tensor")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    k = min(int(top_k), logits.numel())
    adjusted = logits.float() / float(temperature)
    values, indices = torch.topk(adjusted, k=k)
    probs = torch.softmax(values, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).item()
    return int(indices[sampled].item())


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


def sample_from_allowed_logits(
    logits: torch.Tensor,
    allowed_ids: list[int],
    temperature: float = 1.0,
    top_k: int | None = None,
    banned_ids: list[int] | None = None,
) -> torch.Tensor:
    """Sample from logits after masking to an explicit allowed token set."""
    if not allowed_ids:
        return sample_from_logits(logits, temperature=temperature, banned_ids=banned_ids)
    adjusted = torch.full_like(logits.float(), -1e9)
    adjusted[torch.as_tensor(allowed_ids, dtype=torch.long)] = logits.float()[torch.as_tensor(allowed_ids, dtype=torch.long)]
    for token_id in banned_ids or []:
        adjusted[int(token_id)] = -1e9
    if top_k is not None and top_k > 0 and top_k < len(allowed_ids):
        values, indices = torch.topk(adjusted, k=top_k)
        top_adjusted = torch.full_like(adjusted, -1e9)
        top_adjusted[indices] = values
        adjusted = top_adjusted
    if temperature <= 0:
        temperature = 1.0
    probs = torch.softmax(adjusted / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(0)


def argmax_from_allowed_logits(
    logits: torch.Tensor,
    allowed_ids: list[int],
    banned_ids: list[int] | None = None,
) -> torch.Tensor:
    """Choose the highest-scoring token from an explicit allowed token set."""
    if not allowed_ids:
        return argmax_from_logits(logits, banned_ids=banned_ids)
    adjusted = torch.full_like(logits.float(), -1e9)
    idx = torch.as_tensor(allowed_ids, dtype=torch.long)
    adjusted[idx] = logits.float()[idx]
    for token_id in banned_ids or []:
        adjusted[int(token_id)] = -1e9
    return torch.argmax(adjusted, dim=-1)
