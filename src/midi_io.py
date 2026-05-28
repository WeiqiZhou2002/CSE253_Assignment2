from __future__ import annotations

from pathlib import Path

import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

from .tokenization import valid_pitch
from .utils import ensure_dir


VOICE_PROGRAMS = {
    "soprano": 52,
    "alto": 52,
    "tenor": 52,
    "bass": 52,
}


def _note_segments(values: np.ndarray) -> list[tuple[int | None, int]]:
    """Collapse repeated notes into (pitch-or-rest, duration-steps) segments."""
    segments: list[tuple[int | None, int]] = []
    i = 0
    while i < len(values):
        value = values[i]
        pitch = int(value) if valid_pitch(value) else None
        j = i + 1
        while j < len(values):
            next_pitch = int(values[j]) if valid_pitch(values[j]) else None
            if next_pitch != pitch:
                break
            j += 1
        segments.append((pitch, j - i))
        i = j
    return segments


def satb_matrix_to_midi(
    sequence: np.ndarray,
    output_path: str | Path,
    grid: float = 0.5,
    tempo: int = 90,
) -> Path:
    """Write a T x 4 SATB pitch matrix as a multi-track MIDI file."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    seq = np.asarray(sequence)
    if seq.ndim != 2 or seq.shape[1] != 4:
        raise ValueError("expected a T x 4 SATB matrix")

    ticks_per_beat = 480
    ticks_per_step = max(1, int(round(ticks_per_beat * grid)))
    midi = MidiFile(ticks_per_beat=ticks_per_beat)

    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage("set_tempo", tempo=bpm2tempo(tempo), time=0))
    tempo_track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(tempo_track)

    for voice_index, voice_name in enumerate(["soprano", "alto", "tenor", "bass"]):
        track = MidiTrack()
        track.append(MetaMessage("track_name", name=voice_name, time=0))
        track.append(Message("program_change", program=VOICE_PROGRAMS[voice_name], channel=voice_index, time=0))
        silence_ticks = 0
        for pitch, duration_steps in _note_segments(seq[:, voice_index]):
            duration_ticks = duration_steps * ticks_per_step
            if pitch is None:
                silence_ticks += duration_ticks
                continue
            velocity = 74 if voice_name in {"soprano", "alto"} else 68
            track.append(Message("note_on", note=int(pitch), velocity=velocity, channel=voice_index, time=silence_ticks))
            track.append(Message("note_off", note=int(pitch), velocity=0, channel=voice_index, time=duration_ticks))
            silence_ticks = 0
        track.append(MetaMessage("end_of_track", time=silence_ticks))
        midi.tracks.append(track)

    midi.save(output_path)
    return output_path


def write_reference_midi(dataset: dict, output_path: str | Path) -> Path:
    """Save the first test chorale as a reference MIDI file."""
    sequence = dataset["test_pitches"][0]
    grid = float(dataset["metadata"]["grid"])
    return satb_matrix_to_midi(sequence, output_path, grid=grid)

