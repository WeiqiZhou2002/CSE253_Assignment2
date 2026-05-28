from __future__ import annotations

from itertools import combinations

import numpy as np

from .tokenization import valid_pitch


VOICE_RANGES = {
    "soprano": (60, 81),
    "alto": (55, 74),
    "tenor": (48, 67),
    "bass": (40, 60),
}

CONSONANT_INTERVALS = {0, 3, 4, 5, 7, 8, 9}
PERFECT_INTERVALS = {0, 7}


def _pitch_or_none(value: int | float) -> int | None:
    return int(value) if valid_pitch(value) else None


def _valid_rows(sequence: np.ndarray) -> list[list[int]]:
    rows: list[list[int]] = []
    for row in np.asarray(sequence):
        pitches = [_pitch_or_none(v) for v in row]
        if all(p is not None for p in pitches):
            rows.append([int(p) for p in pitches if p is not None])
    return rows


def voice_range_violation_rate(sequence: np.ndarray, voice_ranges: dict[str, tuple[int, int]] | None = None) -> float:
    """Return the fraction of notes outside approximate human SATB ranges."""
    ranges = voice_ranges or VOICE_RANGES
    total = 0
    bad = 0
    for voice_index, voice_name in enumerate(["soprano", "alto", "tenor", "bass"]):
        low, high = ranges[voice_name]
        for value in np.asarray(sequence)[:, voice_index]:
            pitch = _pitch_or_none(value)
            if pitch is None:
                continue
            total += 1
            bad += int(pitch < low or pitch > high)
    return bad / total if total else 0.0


def voice_crossing_count(sequence: np.ndarray) -> int:
    """Count time steps where SATB ordering is violated."""
    count = 0
    for row in _valid_rows(sequence):
        soprano, alto, tenor, bass = row
        if alto > soprano:
            count += 1
        if tenor > alto:
            count += 1
        if bass > tenor:
            count += 1
    return count


def consonance_rate(sequence: np.ndarray, strong_beat_mask: np.ndarray | None = None) -> float:
    """Estimate the fraction of vertical intervals that are consonant."""
    seq = np.asarray(sequence)
    total = 0
    consonant = 0
    if strong_beat_mask is None:
        mask = np.ones(seq.shape[0], dtype=bool)
    else:
        mask = np.asarray(strong_beat_mask, dtype=bool)
    for row, keep in zip(seq, mask):
        if not keep:
            continue
        pitches = [_pitch_or_none(v) for v in row]
        pitches = [p for p in pitches if p is not None]
        for a, b in combinations(pitches, 2):
            total += 1
            consonant += int(abs(a - b) % 12 in CONSONANT_INTERVALS)
    return consonant / total if total else 0.0


def repeated_note_ratio(sequence: np.ndarray) -> float:
    """Return the fraction of adjacent same-voice notes that repeat."""
    seq = np.asarray(sequence)
    repeated = 0
    total = 0
    for voice_index in range(seq.shape[1]):
        previous = None
        for value in seq[:, voice_index]:
            pitch = _pitch_or_none(value)
            if pitch is None:
                previous = None
                continue
            if previous is not None:
                total += 1
                repeated += int(previous == pitch)
            previous = pitch
    return repeated / total if total else 0.0


def chord_diversity(sequence: np.ndarray) -> float:
    """Return unique SATB sonority ratio."""
    rows = [tuple(row) for row in _valid_rows(sequence)]
    return len(set(rows)) / len(rows) if rows else 0.0


def parallel_fifths_octaves(sequence: np.ndarray) -> int:
    """Heuristically count parallel perfect fifths and octaves."""
    rows = _valid_rows(sequence)
    count = 0
    for prev, curr in zip(rows[:-1], rows[1:]):
        for i, j in combinations(range(4), 2):
            prev_interval = abs(prev[i] - prev[j]) % 12
            curr_interval = abs(curr[i] - curr[j]) % 12
            if prev_interval not in PERFECT_INTERVALS or curr_interval not in PERFECT_INTERVALS:
                continue
            motion_i = curr[i] - prev[i]
            motion_j = curr[j] - prev[j]
            if motion_i == 0 or motion_j == 0:
                continue
            if (motion_i > 0 and motion_j > 0) or (motion_i < 0 and motion_j < 0):
                count += 1
    return count


def pitch_class_histogram(sequence: np.ndarray) -> np.ndarray:
    """Return a normalized 12-bin pitch-class histogram."""
    hist = np.zeros(12, dtype=np.float64)
    for value in np.asarray(sequence).reshape(-1):
        pitch = _pitch_or_none(value)
        if pitch is not None:
            hist[pitch % 12] += 1.0
    total = hist.sum()
    return hist / total if total else hist


def histogram_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Return L1 distance between normalized pitch-class histograms."""
    return float(np.abs(np.asarray(hist_a) - np.asarray(hist_b)).sum())


def cadence_quality_heuristic(sequence: np.ndarray) -> float:
    """Score whether the ending resembles a stable tonic sonority."""
    rows = _valid_rows(sequence)
    if not rows:
        return 0.0
    last = rows[-1]
    tonic_pc = last[3] % 12
    pcs = {p % 12 for p in last}
    score = 0.0
    score += 0.4 if tonic_pc in pcs else 0.0
    score += 0.3 if (tonic_pc + 4) % 12 in pcs or (tonic_pc + 3) % 12 in pcs else 0.0
    score += 0.3 if (tonic_pc + 7) % 12 in pcs else 0.0
    return score

