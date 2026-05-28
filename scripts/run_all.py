from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataset, load_processed_dataset, save_processed_dataset
from src.midi_io import write_reference_midi
from src.plots import plot_pitch_class_histogram, plot_sequence_lengths, plot_voice_ranges
from src.train_conditioned import run as run_conditioned
from src.train_unconditioned import run as run_unconditioned
from src.utils import ensure_project_dirs, load_config

from scripts.export_notebook import build_and_export
from scripts.make_submission import make_submission


def build_or_refresh_dataset(config: dict, force: bool = False) -> dict:
    path = ROOT / config["paths"]["processed_data"]
    if path.exists() and not force:
        return load_processed_dataset(path)
    dataset = build_dataset(seed=config["seed"], vocab_config=config["vocab"], **config["data"])
    save_processed_dataset(dataset, path)
    return dataset


def write_eda_outputs(config: dict, dataset: dict) -> None:
    figures_dir = ROOT / config["paths"]["figures_dir"]
    tables_dir = ROOT / config["paths"]["tables_dir"]
    midi_dir = ROOT / config["paths"]["midi_dir"]
    all_pitches = dataset["train_pitches"] + dataset["val_pitches"] + dataset["test_pitches"]
    plot_sequence_lengths([len(seq) for seq in all_pitches], figures_dir / "sequence_lengths.png")
    plot_voice_ranges(all_pitches, figures_dir / "voice_ranges.png")
    plot_pitch_class_histogram(all_pitches, figures_dir / "pitch_class_histogram.png")
    write_reference_midi(dataset, midi_dir / "reference_test_chorale.mid")

    summary = pd.DataFrame(
        [
            {"split": "train", "sequences": len(dataset["train"]), "avg_steps": sum(map(len, dataset["train"])) / len(dataset["train"])},
            {"split": "validation", "sequences": len(dataset["val"]), "avg_steps": sum(map(len, dataset["val"])) / len(dataset["val"])},
            {"split": "test", "sequences": len(dataset["test"]), "avg_steps": sum(map(len, dataset["test"])) / len(dataset["test"])},
        ]
    )
    summary.to_csv(tables_dir / "dataset_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    ensure_project_dirs(config)
    dataset = build_or_refresh_dataset(config, force=args.force_data)
    write_eda_outputs(config, dataset)
    run_unconditioned(config)
    run_conditioned(config)
    build_and_export(config)
    make_submission(config)
    print("Done. Submission artifacts are in submission/.")


if __name__ == "__main__":
    main()

