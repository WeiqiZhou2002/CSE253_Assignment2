from __future__ import annotations

from typing import Any

import numpy as np

from .metrics_music import VOICE_RANGES
from .tokenization import valid_pitch


VOICE_NAMES = ["soprano", "alto", "tenor", "bass"]


def allowed_pitch_ids(token_to_id: dict[Any, int]) -> dict[str, list[int]]:
    """Return allowed token ids for each SATB voice range."""
    allowed: dict[str, list[int]] = {}
    for voice, (low, high) in VOICE_RANGES.items():
        allowed[voice] = [int(token_to_id[pitch]) for pitch in range(low, high + 1) if pitch in token_to_id]
    return allowed


def _nearest_pitch_for_pc(
    pc: int,
    target: int,
    low: int,
    high: int,
    not_equal: int | None = None,
) -> int:
    candidates = [pc + 12 * octave for octave in range(0, 11) if low <= pc + 12 * octave <= high]
    if not_equal is not None:
        candidates = [pitch for pitch in candidates if pitch != not_equal]
    if not candidates:
        center = int(round((low + high) / 2))
        return min(max(center, low), high)
    return min(candidates, key=lambda pitch: (abs(pitch - target), abs(pitch - (low + high) / 2)))


def _fit_pitch(value: int, voice: str, target: int | None = None, not_equal: int | None = None) -> int:
    low, high = VOICE_RANGES[voice]
    pitch = int(value) if valid_pitch(value) else int(round((low + high) / 2))
    return _nearest_pitch_for_pc(pitch % 12, target if target is not None else pitch, low, high, not_equal)


def enforce_satb_ranges_and_order(sequence: np.ndarray) -> np.ndarray:
    """Clip pitches into voice ranges and repair obvious voice crossings."""
    seq = np.asarray(sequence).copy()
    repaired = np.zeros_like(seq)
    for t, row in enumerate(seq):
        soprano = _fit_pitch(int(row[0]), "soprano")
        alto_high = min(VOICE_RANGES["alto"][1], soprano - 2)
        alto = _nearest_pitch_for_pc(int(row[1]) % 12, int(row[1]), VOICE_RANGES["alto"][0], max(VOICE_RANGES["alto"][0], alto_high))

        tenor_high = min(VOICE_RANGES["tenor"][1], alto - 2)
        tenor = _nearest_pitch_for_pc(int(row[2]) % 12, int(row[2]), VOICE_RANGES["tenor"][0], max(VOICE_RANGES["tenor"][0], tenor_high))

        bass_high = min(VOICE_RANGES["bass"][1], tenor - 2)
        bass = _nearest_pitch_for_pc(int(row[3]) % 12, int(row[3]), VOICE_RANGES["bass"][0], max(VOICE_RANGES["bass"][0], bass_high))
        repaired[t] = [soprano, alto, tenor, bass]
    return repaired


def repair_stagnant_lower_voices(
    model_tokens: np.ndarray,
    baseline_tokens: np.ndarray,
    max_repeat_steps: int = 8,
) -> np.ndarray:
    """Use baseline tokens to repair very long repeated lower-voice runs."""
    repaired = np.asarray(model_tokens).copy()
    baseline = np.asarray(baseline_tokens)
    for voice in range(1, 4):
        run_start = 0
        while run_start < len(repaired):
            run_end = run_start + 1
            while run_end < len(repaired) and repaired[run_end, voice] == repaired[run_start, voice]:
                run_end += 1
            if run_end - run_start > max_repeat_steps:
                repaired[run_start + max_repeat_steps : run_end, voice] = baseline[run_start + max_repeat_steps : run_end, voice]
            run_start = run_end
    return repaired


def add_inner_voice_motion(
    sequence: np.ndarray,
    max_repeat_steps: int = 6,
) -> np.ndarray:
    """Break long lower-voice pitch runs with nearby chord tones."""
    seq = enforce_satb_ranges_and_order(sequence)
    for voice_index, voice in enumerate(VOICE_NAMES[1:], start=1):
        run_start = 0
        while run_start < len(seq):
            run_end = run_start + 1
            while run_end < len(seq) and seq[run_end, voice_index] == seq[run_start, voice_index]:
                run_end += 1
            if run_end - run_start > max_repeat_steps:
                for t in range(run_start + max_repeat_steps, run_end, max_repeat_steps):
                    chord_pcs = [int(p) % 12 for p in seq[t] if valid_pitch(p)]
                    current = int(seq[t, voice_index])
                    lower_voice = VOICE_NAMES[voice_index]
                    candidates = [
                        _fit_pitch(pc, lower_voice, target=current, not_equal=current)
                        for pc in chord_pcs
                        if _fit_pitch(pc, lower_voice, target=current, not_equal=current) != current
                    ]
                    if candidates:
                        seq[t, voice_index] = min(candidates, key=lambda p: abs(p - current))
        # Re-scan after each voice; this keeps crossings repaired after inserted motion.
            run_start = run_end
        seq = enforce_satb_ranges_and_order(seq)
    return seq


def force_simple_cadence(sequence: np.ndarray) -> np.ndarray:
    """End with a stable closed-position major sonority in the final bar."""
    seq = enforce_satb_ranges_and_order(sequence)
    if len(seq) < 4:
        return seq
    bass_pc = int(seq[-1, 3]) % 12
    final = [
        _nearest_pitch_for_pc((bass_pc + 7) % 12, int(seq[-1, 0]), *VOICE_RANGES["soprano"]),
        _nearest_pitch_for_pc((bass_pc + 4) % 12, int(seq[-1, 1]), *VOICE_RANGES["alto"]),
        _nearest_pitch_for_pc(bass_pc, int(seq[-1, 2]), *VOICE_RANGES["tenor"]),
        _nearest_pitch_for_pc(bass_pc, int(seq[-1, 3]), *VOICE_RANGES["bass"]),
    ]
    seq[-4:] = np.asarray(final, dtype=np.int64)
    return enforce_satb_ranges_and_order(seq)


def polish_generated_sequence(sequence: np.ndarray, max_repeat_steps: int = 6, force_cadence: bool = True) -> np.ndarray:
    """Apply light musical repairs to generated SATB pitch matrices."""
    polished = enforce_satb_ranges_and_order(sequence)
    polished = add_inner_voice_motion(polished, max_repeat_steps=max_repeat_steps)
    if force_cadence:
        polished = force_simple_cadence(polished)
    return polished
