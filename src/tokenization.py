from __future__ import annotations

from typing import Any

import numpy as np


PAD_VALUE = -99
REST_VALUE = -1
HOLD_VALUE = -2
START_VALUE = -3
END_VALUE = -4

SPECIAL_TOKENS = ["PAD", "REST", "HOLD", "START", "END"]
SPECIAL_TO_VALUE = {
    "PAD": PAD_VALUE,
    "REST": REST_VALUE,
    "HOLD": HOLD_VALUE,
    "START": START_VALUE,
    "END": END_VALUE,
}
VALUE_TO_SPECIAL = {v: k for k, v in SPECIAL_TO_VALUE.items()}


def build_vocab(
    min_pitch: int = 21,
    max_pitch: int = 108,
    special_tokens: list[str] | None = None,
) -> tuple[dict[Any, int], dict[int, Any]]:
    """Return token-to-id and id-to-token dictionaries."""
    specials = special_tokens or SPECIAL_TOKENS
    token_to_id: dict[Any, int] = {}
    for tok in specials:
        token_to_id[tok] = len(token_to_id)
    for pitch in range(min_pitch, max_pitch + 1):
        token_to_id[pitch] = len(token_to_id)
    id_to_token = {idx: tok for tok, idx in token_to_id.items()}
    return token_to_id, id_to_token


def token_id(token_to_id: dict[Any, int], token: Any) -> int:
    """Map a pitch or special symbol to an integer id."""
    if isinstance(token, np.integer):
        token = int(token)
    if isinstance(token, str):
        return token_to_id[token]
    if token in VALUE_TO_SPECIAL:
        return token_to_id[VALUE_TO_SPECIAL[int(token)]]
    if int(token) in token_to_id:
        return token_to_id[int(token)]
    return token_to_id["REST"]


def encode_pitch_matrix(matrix: np.ndarray, token_to_id: dict[Any, int]) -> np.ndarray:
    """Convert a T x 4 pitch/special matrix to token ids."""
    arr = np.asarray(matrix)
    out = np.full(arr.shape, token_to_id["PAD"], dtype=np.int64)
    for idx, value in np.ndenumerate(arr):
        out[idx] = token_id(token_to_id, value)
    return out


def decode_token_matrix(tokens: np.ndarray, id_to_token: dict[int, Any]) -> np.ndarray:
    """Convert token ids back to MIDI pitches and negative special values."""
    arr = np.asarray(tokens)
    out = np.full(arr.shape, PAD_VALUE, dtype=np.int64)
    for idx, value in np.ndenumerate(arr):
        token = id_to_token[int(value)]
        if isinstance(token, str):
            out[idx] = SPECIAL_TO_VALUE[token]
        else:
            out[idx] = int(token)
    return out


def decode_token_sequence(tokens: np.ndarray, id_to_token: dict[int, Any]) -> list[Any]:
    """Decode a one-dimensional token sequence."""
    return [id_to_token[int(t)] for t in np.asarray(tokens).reshape(-1)]


def valid_pitch(value: Any) -> bool:
    """Return True for normal MIDI pitches."""
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False


def special_id(token_to_id: dict[Any, int], name: str) -> int:
    """Return the id for a special token."""
    return int(token_to_id[name])

