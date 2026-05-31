from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataset, load_processed_dataset, save_processed_dataset
from src.midi_io import satb_matrix_to_midi
from src.tokenization import REST_VALUE, decode_token_sequence
from src.train_conditioned import make_custom_soprano_tokens
from src.utils import ensure_dir, load_config


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_name(pitch: int) -> str:
    octave = pitch // 12 - 1
    return f"{NOTE_NAMES[pitch % 12]}{octave}"


def load_or_build_dataset(config: dict) -> dict:
    path = ROOT / config["paths"]["processed_data"]
    if path.exists():
        return load_processed_dataset(path)
    dataset = build_dataset(seed=config["seed"], vocab_config=config["vocab"], **config["data"])
    save_processed_dataset(dataset, path)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    dataset = load_or_build_dataset(config)
    token_to_id = dataset["token_to_id"]
    id_to_token = dataset["id_to_token"]
    length = int(config["conditioned_model"].get("sample_length", 96))
    grid = float(dataset["metadata"]["grid"])

    soprano_tokens = make_custom_soprano_tokens(token_to_id, max_steps=length)
    decoded = decode_token_sequence(soprano_tokens, id_to_token)
    pitches = [int(p) for p in decoded if isinstance(p, int)]

    midi_dir = ensure_dir(ROOT / config["paths"]["midi_dir"])
    tables_dir = ensure_dir(ROOT / config["paths"]["tables_dir"])

    melody_matrix = np.full((len(pitches), 4), REST_VALUE, dtype=np.int64)
    melody_matrix[:, 0] = np.asarray(pitches, dtype=np.int64)
    midi_path = midi_dir / "conditioned_soprano_melody.mid"
    csv_path = tables_dir / "conditioned_soprano_melody.csv"
    txt_path = tables_dir / "conditioned_soprano_melody.txt"

    satb_matrix_to_midi(melody_matrix, midi_path, grid=grid, max_note_steps=4)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "quarter_time", "midi_pitch", "note"])
        writer.writeheader()
        for step, pitch in enumerate(pitches):
            writer.writerow(
                {
                    "step": step,
                    "quarter_time": step * grid,
                    "midi_pitch": pitch,
                    "note": midi_name(pitch),
                }
            )

    compact = []
    previous = None
    duration_steps = 0
    for pitch in pitches:
        if previous is None or pitch == previous:
            previous = pitch
            duration_steps += 1
            continue
        compact.append((previous, duration_steps))
        previous = pitch
        duration_steps = 1
    if previous is not None:
        compact.append((previous, duration_steps))

    lines = [
        "Conditioned-task soprano melody",
        f"grid_quarter_length: {grid}",
        "format: note(duration_in_quarter_lengths)",
        "",
        " ".join(f"{midi_name(pitch)}({duration_steps * grid:g})" for pitch, duration_steps in compact),
        "",
        "full_midi_pitches_by_quarter_note:",
        " ".join(str(pitch) for pitch in pitches[::2]),
        "",
        "compact_midi_pitches:",
        " ".join(str(pitch) for pitch, _ in compact),
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {midi_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
