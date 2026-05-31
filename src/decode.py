from __future__ import annotations

from itertools import product
from typing import Any, Callable

import numpy as np
import torch

from .evaluate import evaluate_music_quality, musical_badness_score
from .metrics_music import CONSONANT_INTERVALS, VOICE_RANGES
from .postprocess import polish_generated_sequence
from .sample import sample_top_k_temperature
from .tokenization import decode_token_matrix, valid_pitch


VOICE_NAMES = ["soprano", "alto", "tenor", "bass"]


def _pitch(token_id: int, id_to_token: dict[int, Any]) -> int | None:
    token = id_to_token[int(token_id)]
    return int(token) if isinstance(token, int) else None


def _top_allowed_log_probs(
    logits: torch.Tensor,
    allowed_ids: list[int],
    top_k: int,
) -> list[tuple[int, float]]:
    adjusted = torch.full_like(logits.float(), -1e9)
    idx = torch.as_tensor(allowed_ids, dtype=torch.long)
    adjusted[idx] = logits.float()[idx]
    log_probs = torch.log_softmax(adjusted, dim=-1)
    values, indices = torch.topk(log_probs, k=min(top_k, len(allowed_ids)))
    return [(int(token_id), float(log_prob)) for token_id, log_prob in zip(indices.tolist(), values.tolist())]


def _parallel_fifth_octave_penalty(
    current: list[int],
    previous: list[int] | None,
    id_to_token: dict[int, Any],
) -> float:
    if previous is None:
        return 0.0
    curr_pitches = [_pitch(token_id, id_to_token) for token_id in current]
    prev_pitches = [_pitch(token_id, id_to_token) for token_id in previous]
    if any(p is None for p in curr_pitches + prev_pitches):
        return 0.0
    penalty = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            prev_interval = abs(int(prev_pitches[i]) - int(prev_pitches[j])) % 12
            curr_interval = abs(int(curr_pitches[i]) - int(curr_pitches[j])) % 12
            if prev_interval not in {0, 7} or curr_interval not in {0, 7}:
                continue
            motion_i = int(curr_pitches[i]) - int(prev_pitches[i])
            motion_j = int(curr_pitches[j]) - int(prev_pitches[j])
            if motion_i == 0 or motion_j == 0:
                continue
            if (motion_i > 0 and motion_j > 0) or (motion_i < 0 and motion_j < 0):
                penalty += 2.0
    return penalty


def chord_penalty(
    current_chord: list[int],
    previous_chord: list[int] | None = None,
    beat_index: int = 0,
    id_to_token: dict[int, Any] | None = None,
    steps_per_measure: int = 8,
) -> float:
    """Penalize bad SATB token combinations for beam-search decoding."""
    if id_to_token is None:
        pitches = [int(v) if valid_pitch(v) else None for v in current_chord]
        prev_pitches = [int(v) if valid_pitch(v) else None for v in previous_chord] if previous_chord else None
    else:
        pitches = [_pitch(token_id, id_to_token) for token_id in current_chord]
        prev_pitches = [_pitch(token_id, id_to_token) for token_id in previous_chord] if previous_chord else None
    if any(p is None for p in pitches):
        return 100.0

    soprano, alto, tenor, bass = [int(p) for p in pitches if p is not None]
    penalty = 0.0
    if not (soprano >= alto >= tenor >= bass):
        penalty += 10.0
    for voice_name, pitch in zip(VOICE_NAMES, [soprano, alto, tenor, bass]):
        low, high = VOICE_RANGES[voice_name]
        if pitch < low or pitch > high:
            penalty += 5.0

    strong_beat = beat_index % steps_per_measure in {0, steps_per_measure // 2}
    if strong_beat:
        intervals = [abs(a - b) % 12 for i, a in enumerate(pitches) for b in pitches[i + 1 :] if b is not None]
        dissonant = sum(interval not in CONSONANT_INTERVALS for interval in intervals)
        if dissonant:
            penalty += 3.0 * dissonant / max(1, len(intervals))

    if prev_pitches is not None and not any(p is None for p in prev_pitches):
        for pitch, prev_pitch in zip(pitches, prev_pitches):
            if abs(int(pitch) - int(prev_pitch)) > 12:
                penalty += 2.0
        if id_to_token is None:
            penalty += _parallel_pitch_penalty([int(p) for p in pitches], [int(p) for p in prev_pitches])
        else:
            penalty += _parallel_fifth_octave_penalty(current_chord, previous_chord, id_to_token)

    if beat_index >= 0 and bass % 12 == 0 and {0, 4, 7}.issubset({p % 12 for p in [soprano, alto, tenor, bass]}):
        penalty -= 1.0
    return float(penalty)


def _parallel_pitch_penalty(current: list[int], previous: list[int]) -> float:
    penalty = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            prev_interval = abs(previous[i] - previous[j]) % 12
            curr_interval = abs(current[i] - current[j]) % 12
            if prev_interval not in {0, 7} or curr_interval not in {0, 7}:
                continue
            motion_i = current[i] - previous[i]
            motion_j = current[j] - previous[j]
            if motion_i and motion_j and ((motion_i > 0 and motion_j > 0) or (motion_i < 0 and motion_j < 0)):
                penalty += 2.0
    return penalty


def harmonize_with_beam_search(
    model: torch.nn.Module,
    soprano_sequence: np.ndarray,
    allowed_ids_by_voice: dict[str, list[int]],
    id_to_token: dict[int, Any],
    device: torch.device,
    beam_size: int = 8,
    top_k_per_voice: int = 5,
    rule_weight: float = 1.0,
    steps_per_measure: int = 8,
) -> np.ndarray:
    """Generate lower voices conditioned on soprano using model score plus SATB penalties."""
    model.eval()
    soprano_tokens = np.asarray(soprano_sequence, dtype=np.int64)
    soprano_tensor = torch.as_tensor(soprano_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        alto_logits, tenor_logits, bass_logits = model(soprano_tensor)

    beams: list[tuple[float, list[list[int]]]] = [(0.0, [])]
    voice_options = [
        ("alto", alto_logits.detach().cpu()[0]),
        ("tenor", tenor_logits.detach().cpu()[0]),
        ("bass", bass_logits.detach().cpu()[0]),
    ]
    for t, soprano_token in enumerate(soprano_tokens):
        options_per_voice = [
            _top_allowed_log_probs(logits[t], allowed_ids_by_voice[voice_name], top_k_per_voice)
            for voice_name, logits in voice_options
        ]
        expanded: list[tuple[float, list[list[int]]]] = []
        for score, states in beams:
            previous = states[-1] if states else None
            for alto, tenor, bass in product(*options_per_voice):
                chord = [int(soprano_token), alto[0], tenor[0], bass[0]]
                neg_log_prob = -(alto[1] + tenor[1] + bass[1])
                penalty = chord_penalty(
                    chord,
                    previous_chord=previous,
                    beat_index=t,
                    id_to_token=id_to_token,
                    steps_per_measure=steps_per_measure,
                )
                expanded.append((score + neg_log_prob + rule_weight * penalty, states + [chord]))
        expanded.sort(key=lambda item: item[0])
        beams = expanded[:beam_size]
    return np.asarray(beams[0][1], dtype=np.int64)


def rerank_generated_candidates(
    candidates: list[np.ndarray],
    id_to_token: dict[int, Any],
    reference_hist: np.ndarray | None,
    steps_per_measure: int = 8,
    max_repeat_steps: int = 6,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Decode, postprocess, score, and select the best generated token candidates."""
    rows: list[dict[str, float]] = []
    best_matrix: np.ndarray | None = None
    best_score = float("inf")
    for candidate_id, tokens in enumerate(candidates):
        matrix = polish_generated_sequence(decode_token_matrix(tokens, id_to_token), max_repeat_steps=max_repeat_steps)
        quality = evaluate_music_quality(matrix, reference_hist=reference_hist, steps_per_measure=steps_per_measure)
        badness = musical_badness_score(quality)
        row = {"candidate": float(candidate_id), "badness": badness, **quality}
        rows.append(row)
        if badness < best_score:
            best_score = badness
            best_matrix = matrix
    if best_matrix is None:
        raise ValueError("no candidates supplied")
    return best_matrix, rows


def generate_and_rerank(
    generate_fn: Callable[[int], np.ndarray],
    id_to_token: dict[int, Any],
    n_candidates: int = 50,
    reference_hist: np.ndarray | None = None,
    steps_per_measure: int = 8,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Generate many token candidates with deterministic seeds and return the best matrix."""
    candidates = []
    for seed in range(n_candidates):
        torch.manual_seed(seed)
        candidates.append(generate_fn(seed))
    return rerank_generated_candidates(candidates, id_to_token, reference_hist, steps_per_measure)

