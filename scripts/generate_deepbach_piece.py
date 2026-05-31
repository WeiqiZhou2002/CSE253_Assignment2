from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_dataset, load_processed_dataset, save_processed_dataset
from src.deepbach_gibbs import DeepBachGibbsSampler, select_melodic_soprano
from src.evaluate import aggregate_pitch_histogram, save_table, sequence_music_metrics
from src.midi_io import satb_matrix_to_midi
from src.postprocess import enforce_satb_ranges_and_order, force_simple_cadence
from src.tokenization import decode_token_matrix, special_id
from src.train_conditioned import make_custom_soprano_tokens
from src.utils import ensure_project_dirs, load_config


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
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--custom-melody", action="store_true")
    parser.add_argument("--replace-final", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    ensure_project_dirs(config)
    dataset = load_or_build_dataset(config)
    token_to_id = dataset["token_to_id"]
    id_to_token = dataset["id_to_token"]
    pad_id = special_id(token_to_id, "PAD")
    steps_per_measure = int(round(4.0 / float(dataset["metadata"]["grid"])))
    length = int(config["conditioned_model"].get("sample_length", 96))

    if args.custom_melody:
        soprano = make_custom_soprano_tokens(token_to_id, max_steps=length)
    else:
        soprano = select_melodic_soprano(dataset["test"], id_to_token, target_length=length)
    sampler = DeepBachGibbsSampler(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_id=pad_id,
        steps_per_measure=steps_per_measure,
        seed=int(config["seed"]),
    ).fit(dataset["train"])

    token_sample = sampler.harmonize(
        soprano,
        iterations=args.iterations,
        temperature=args.temperature,
        candidate_limit=100,
    )
    pitch_sample = force_simple_cadence(enforce_satb_ranges_and_order(decode_token_matrix(token_sample, id_to_token)))

    midi_dir = ROOT / config["paths"]["midi_dir"]
    tables_dir = ROOT / config["paths"]["tables_dir"]
    deepbach_path = midi_dir / "deepbach_conditioned.mid"
    satb_matrix_to_midi(pitch_sample, deepbach_path, grid=float(dataset["metadata"]["grid"]), max_note_steps=4)

    ref_hist = aggregate_pitch_histogram(dataset["train_pitches"])
    row = {
        "model": "DeepBach-inspired Gibbs",
        "alto_accuracy": "",
        "tenor_accuracy": "",
        "bass_accuracy": "",
        "average_voice_accuracy": "",
        "exact_atb_accuracy": "",
        **sequence_music_metrics(pitch_sample, ref_hist, steps_per_measure),
        "notes": "Gibbs sampler over Bach-trained local conditional distributions",
    }
    save_table([row], tables_dir / "deepbach_conditioned_metrics.csv")

    if args.replace_final:
        final_output = midi_dir / "symbolic_conditioned.mid"
        shutil.copy2(deepbach_path, final_output)
        submission_output = ROOT / "submission" / "symbolic_conditioned.mid"
        if submission_output.parent.exists():
            shutil.copy2(deepbach_path, submission_output)
        print(f"Wrote {deepbach_path} and replaced final conditioned MIDI.")
    else:
        print(f"Wrote {deepbach_path}.")
    print(f"Dataset source: {dataset['metadata']['source']} ({dataset['metadata']['source_note']})")


if __name__ == "__main__":
    main()
