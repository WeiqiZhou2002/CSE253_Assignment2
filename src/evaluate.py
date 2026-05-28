from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics_music import (
    cadence_quality_heuristic,
    chord_diversity,
    consonance_rate,
    histogram_distance,
    parallel_fifths_octaves,
    pitch_class_histogram,
    repeated_note_ratio,
    voice_crossing_count,
    voice_range_violation_rate,
)
from .utils import ensure_dir


def strong_beat_mask(length: int, steps_per_measure: int = 8) -> np.ndarray:
    """Return a mask for beats 1 and 3 in 4/4 at eighth-note resolution."""
    return np.array([(t % steps_per_measure) in {0, steps_per_measure // 2} for t in range(length)], dtype=bool)


def sequence_music_metrics(
    sequence: np.ndarray,
    reference_hist: np.ndarray | None = None,
    steps_per_measure: int = 8,
) -> dict[str, float]:
    """Compute music-theory-inspired metrics for one SATB sequence."""
    mask = strong_beat_mask(len(sequence), steps_per_measure=steps_per_measure)
    hist = pitch_class_histogram(sequence)
    metrics = {
        "range_violations": voice_range_violation_rate(sequence),
        "voice_crossings": float(voice_crossing_count(sequence)),
        "strong_beat_consonance": consonance_rate(sequence, strong_beat_mask=mask),
        "repeated_note_ratio": repeated_note_ratio(sequence),
        "chord_diversity": chord_diversity(sequence),
        "parallel_fifths_octaves": float(parallel_fifths_octaves(sequence)),
        "cadence_score": cadence_quality_heuristic(sequence),
    }
    if reference_hist is not None:
        metrics["pitch_class_hist_distance"] = histogram_distance(hist, reference_hist)
    return metrics


def aggregate_pitch_histogram(sequences: list[np.ndarray]) -> np.ndarray:
    """Aggregate a normalized pitch-class histogram over many sequences."""
    hist = np.zeros(12, dtype=np.float64)
    for seq in sequences:
        hist += pitch_class_histogram(seq)
    total = hist.sum()
    return hist / total if total else hist


def aggregate_music_metrics(
    sequences: list[np.ndarray],
    reference_hist: np.ndarray | None = None,
    steps_per_measure: int = 8,
) -> dict[str, float]:
    """Average music metrics across a collection of sequences."""
    rows = [sequence_music_metrics(seq, reference_hist, steps_per_measure) for seq in sequences]
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def save_table(rows: list[dict[str, Any]], path: str | Path) -> pd.DataFrame:
    """Save a list of metric dictionaries as CSV and return the DataFrame."""
    path = Path(path)
    ensure_dir(path.parent)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df

