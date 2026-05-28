from __future__ import annotations

import math
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

from .tokenization import REST_VALUE, build_vocab, encode_pitch_matrix
from .utils import ensure_dir, set_seed


VOICE_ORDER = ["soprano", "alto", "tenor", "bass"]


def load_bach_chorales(max_chorales: int | None = None) -> list[Any]:
    """Load Bach chorales from music21 if the dependency is available."""
    try:
        from music21 import corpus
    except ImportError as exc:
        raise RuntimeError("music21 is not installed") from exc

    chorales = list(corpus.chorales.Iterator())
    if max_chorales is not None:
        chorales = chorales[:max_chorales]
    return chorales


def _pitch_from_music21_element(element: Any, voice_index: int) -> int:
    """Extract a representative MIDI pitch from a music21 note/chord/rest."""
    if getattr(element, "isRest", False):
        return REST_VALUE
    if getattr(element, "isNote", False):
        return int(element.pitch.midi)
    if getattr(element, "isChord", False):
        pitches = sorted(int(p.midi) for p in element.pitches)
        if not pitches:
            return REST_VALUE
        if voice_index == 0:
            return pitches[-1]
        if voice_index == 3:
            return pitches[0]
        return pitches[min(voice_index, len(pitches) - 1)]
    return REST_VALUE


def score_to_satb_matrix(score: Any, grid: float = 0.5) -> np.ndarray:
    """Convert one music21 score into a T x 4 SATB MIDI-pitch matrix."""
    parts = list(getattr(score, "parts", []))
    if len(parts) < 4:
        raise ValueError("score has fewer than four parts")
    parts = parts[:4]
    total_quarters = max(float(getattr(part, "highestTime", 0.0)) for part in parts)
    if total_quarters <= 0:
        raise ValueError("score has no duration")

    steps = max(1, int(math.ceil(total_quarters / grid)))
    matrix = np.full((steps, 4), REST_VALUE, dtype=np.int64)

    for voice_index, part in enumerate(parts):
        for element in part.recurse().notesAndRests:
            offset = float(element.offset)
            duration = float(element.duration.quarterLength)
            start = max(0, int(round(offset / grid)))
            end = max(start + 1, int(round((offset + duration) / grid)))
            if start >= steps:
                continue
            pitch = _pitch_from_music21_element(element, voice_index)
            matrix[start : min(end, steps), voice_index] = pitch
    return matrix


def _nearest_pitch_for_pc(pc: int, previous: int, low: int, high: int) -> int:
    candidates = [pc + 12 * octave for octave in range(0, 11) if low <= pc + 12 * octave <= high]
    if not candidates:
        return min(max(previous, low), high)
    return min(candidates, key=lambda p: (abs(p - previous), abs(p - (low + high) / 2)))


def _add_neighbor_motion(note: int, low: int, high: int, key_pcs: set[int], rng: random.Random) -> int:
    if rng.random() > 0.28:
        return note
    for step in rng.sample([-2, 2, -1, 1], k=4):
        candidate = note + step
        if low <= candidate <= high and candidate % 12 in key_pcs:
            return candidate
    return note


def _make_synthetic_chorale(index: int, grid: float, rng: random.Random) -> np.ndarray:
    """Create one deterministic SATB chorale-style matrix for fallback runs."""
    progressions = [
        ["I", "V", "vi", "IV", "ii", "V", "I", "V", "I", "IV", "V", "vi", "ii", "V", "I", "I"],
        ["I", "IV", "I", "V", "vi", "ii", "V", "I", "IV", "V", "I", "vi", "ii", "V", "I", "I"],
        ["I", "vi", "IV", "V", "I", "ii", "V", "I", "vi", "IV", "ii", "V", "I", "V", "I", "I"],
        ["I", "V", "I", "IV", "ii", "V", "I", "vi", "IV", "I", "V", "vi", "ii", "V", "I", "I"],
    ]
    chord_degrees = {
        "I": [0, 4, 7],
        "ii": [2, 5, 9],
        "IV": [5, 9, 0],
        "V": [7, 11, 2],
        "vi": [9, 0, 4],
    }
    key_offsets = [-5, -2, 0, 2, 5, 7]
    key_offset = key_offsets[index % len(key_offsets)]
    key_pc = key_offset % 12
    major_scale = {(key_pc + pc) % 12 for pc in [0, 2, 4, 5, 7, 9, 11]}
    progression = list(progressions[index % len(progressions)])
    rng.shuffle(progression)
    progression[-2:] = ["V", "I"]

    steps_per_chord = max(1, int(round(2.0 / grid)))
    matrix = np.zeros((len(progression) * steps_per_chord, 4), dtype=np.int64)
    previous = [72 + key_offset, 64 + key_offset, 55 + key_offset, 48 + key_offset]
    ranges = [(60, 81), (55, 74), (48, 67), (40, 60)]

    for chord_idx, chord_name in enumerate(progression):
        pcs = [(key_pc + degree) % 12 for degree in chord_degrees[chord_name]]
        root_pc = pcs[0]
        bass = _nearest_pitch_for_pc(root_pc, previous[3], 40, 55)
        tenor = _nearest_pitch_for_pc(rng.choice(pcs), previous[2], 50, 64)
        alto = _nearest_pitch_for_pc(rng.choice(pcs), previous[1], 58, 71)
        soprano = _nearest_pitch_for_pc(rng.choice(pcs), previous[0], 65, 79)

        voices = [soprano, alto, tenor, bass]
        if voices[0] <= voices[1]:
            voices[0] += 12
        if voices[1] <= voices[2]:
            voices[1] += 12
        if voices[2] <= voices[3]:
            voices[2] += 12
        voices = [min(max(v, lo), hi) for v, (lo, hi) in zip(voices, ranges)]

        start = chord_idx * steps_per_chord
        for local_step in range(steps_per_chord):
            row = voices.copy()
            if local_step % 2 == 1 and chord_idx < len(progression) - 1:
                row[0] = _add_neighbor_motion(row[0], *ranges[0], major_scale, rng)
                row[1] = _add_neighbor_motion(row[1], *ranges[1], major_scale, rng)
            matrix[start + local_step] = row
        previous = voices

    tonic = key_pc
    matrix[-steps_per_chord:, 0] = _nearest_pitch_for_pc((tonic + 7) % 12, previous[0], 67, 79)
    matrix[-steps_per_chord:, 1] = _nearest_pitch_for_pc((tonic + 4) % 12, previous[1], 58, 71)
    matrix[-steps_per_chord:, 2] = _nearest_pitch_for_pc(tonic, previous[2], 50, 64)
    matrix[-steps_per_chord:, 3] = _nearest_pitch_for_pc(tonic, previous[3], 40, 55)
    return matrix


def synthetic_chorale_dataset(count: int = 96, grid: float = 0.5, seed: int = 42) -> list[np.ndarray]:
    """Return a deterministic fallback corpus of chorale-style SATB matrices."""
    rng = random.Random(seed)
    return [_make_synthetic_chorale(i, grid, rng) for i in range(count)]


def _split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train = indices[:n_train].tolist()
    val = indices[n_train : n_train + n_val].tolist()
    test = indices[n_train + n_val :].tolist()
    if not test:
        test = val[-1:]
        val = val[:-1] or train[-1:]
    return train, val, test


def build_dataset(
    grid: float = 0.5,
    max_seq_len: int = 128,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_chorales: int | None = None,
    fallback_chorales: int = 96,
    synthetic_only: bool = False,
    seed: int = 42,
    vocab_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build tokenized train/validation/test splits plus metadata."""
    del test_ratio
    set_seed(seed)
    matrices: list[np.ndarray] = []
    source = ""
    source_note = ""

    if not synthetic_only:
        try:
            scores = load_bach_chorales(max_chorales=max_chorales)
            for score in scores:
                try:
                    matrix = score_to_satb_matrix(score, grid=grid)
                    if matrix.shape[0] >= 8:
                        matrices.append(matrix[:max_seq_len])
                except Exception:
                    continue
            if matrices:
                source = "music21 Bach chorales"
                source_note = f"Loaded {len(matrices)} usable chorales from music21."
        except Exception as exc:
            source_note = f"music21 corpus unavailable ({exc}); used deterministic fallback corpus."

    if len(matrices) < 10:
        matrices = synthetic_chorale_dataset(count=fallback_chorales, grid=grid, seed=seed)
        matrices = [m[:max_seq_len] for m in matrices]
        source = "deterministic chorale-style fallback corpus"
        if not source_note:
            source_note = "Used deterministic fallback corpus."

    vocab_config = vocab_config or {}
    token_to_id, id_to_token = build_vocab(
        min_pitch=int(vocab_config.get("min_pitch", 21)),
        max_pitch=int(vocab_config.get("max_pitch", 108)),
        special_tokens=vocab_config.get("special_tokens"),
    )
    tokens = [encode_pitch_matrix(matrix, token_to_id) for matrix in matrices]
    train_idx, val_idx, test_idx = _split_indices(len(tokens), train_ratio, val_ratio, seed)

    def pick(items: list[Any], idxs: list[int]) -> list[Any]:
        return [items[i] for i in idxs]

    dataset = {
        "train": pick(tokens, train_idx),
        "val": pick(tokens, val_idx),
        "test": pick(tokens, test_idx),
        "train_pitches": pick(matrices, train_idx),
        "val_pitches": pick(matrices, val_idx),
        "test_pitches": pick(matrices, test_idx),
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "metadata": {
            "source": source,
            "source_note": source_note,
            "grid": grid,
            "max_seq_len": max_seq_len,
            "voice_order": VOICE_ORDER,
            "num_sequences": len(tokens),
            "num_train": len(train_idx),
            "num_val": len(val_idx),
            "num_test": len(test_idx),
            "vocab_size": len(token_to_id),
        },
    }
    return dataset


def save_processed_dataset(dataset: dict[str, Any], path: str | Path) -> None:
    """Save processed arrays and metadata."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)


def load_processed_dataset(path: str | Path) -> dict[str, Any]:
    """Load processed arrays and metadata."""
    with open(path, "rb") as f:
        return pickle.load(f)
