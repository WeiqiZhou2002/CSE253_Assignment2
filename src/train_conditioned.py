from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .baselines import LookupHarmonizationBaseline
from .data import build_dataset, load_processed_dataset, save_processed_dataset
from .decode import harmonize_with_beam_search
from .evaluate import aggregate_pitch_histogram, save_table, sequence_music_metrics
from .midi_io import satb_matrix_to_midi
from .models import SopranoConditionedHarmonizer
from .plots import plot_metric_comparison, plot_training_curve
from .postprocess import allowed_pitch_ids, polish_generated_sequence
from .sample import sample_from_allowed_logits
from .tokenization import decode_token_matrix, special_id
from .utils import PROJECT_ROOT, batched, ensure_project_dirs, load_config, resolve_device, save_json, set_seed


def _load_or_build_dataset(config: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / config["paths"]["processed_data"]
    if path.exists():
        return load_processed_dataset(path)
    dataset = build_dataset(seed=config["seed"], vocab_config=config["vocab"], **config["data"])
    save_processed_dataset(dataset, path)
    return dataset


def _make_harmonizer_batch(
    sequences: list[np.ndarray],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    soprano = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    target = torch.full((len(sequences), max_len, 3), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        n = len(seq)
        arr = torch.as_tensor(seq, dtype=torch.long)
        soprano[i, :n] = arr[:, 0]
        target[i, :n] = arr[:, 1:4]
    return soprano.to(device), target.to(device)


def _harmonizer_loss(outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    loss = torch.zeros((), device=target.device)
    for voice_index, logits in enumerate(outputs):
        loss = loss + criterion(logits.reshape(-1, logits.shape[-1]), target[:, :, voice_index].reshape(-1))
    return loss


def _evaluate_model(
    model: SopranoConditionedHarmonizer,
    sequences: list[np.ndarray],
    pad_id: int,
    banned_ids: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    losses: list[float] = []
    correct = np.zeros(3, dtype=np.float64)
    total = 0.0
    exact = 0.0
    with torch.no_grad():
        for batch in batched(sequences, batch_size):
            soprano, target = _make_harmonizer_batch(batch, pad_id, device)
            outputs = model(soprano)
            losses.append(float(_harmonizer_loss(outputs, target, criterion).detach().cpu()))
            predictions = []
            for logits in outputs:
                adjusted = logits.detach().cpu()
                for token_id in banned_ids:
                    adjusted[:, :, token_id] = -1e9
                predictions.append(torch.argmax(adjusted, dim=-1))
            pred = torch.stack(predictions, dim=-1).numpy()
            true = target.detach().cpu().numpy()
            mask = ~np.any(true == pad_id, axis=2)
            for row_pred, row_true, keep in zip(pred.reshape(-1, 3), true.reshape(-1, 3), mask.reshape(-1)):
                if not keep:
                    continue
                matches = row_pred == row_true
                correct += matches.astype(np.float64)
                exact += float(np.all(matches))
                total += 1.0
    total = max(total, 1.0)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "alto_accuracy": float(correct[0] / total),
        "tenor_accuracy": float(correct[1] / total),
        "bass_accuracy": float(correct[2] / total),
        "average_voice_accuracy": float(correct.mean() / total),
        "exact_atb_accuracy": float(exact / total),
    }


def generate_harmonization(
    model: SopranoConditionedHarmonizer,
    soprano_tokens: np.ndarray,
    banned_ids: list[int],
    allowed_ids_by_voice: dict[str, list[int]],
    device: torch.device,
    temperature: float = 0.9,
) -> np.ndarray:
    """Generate ATB token predictions for a fixed soprano melody."""
    model.eval()
    soprano = torch.as_tensor(soprano_tokens, dtype=torch.long, device=device).unsqueeze(0)
    output = np.full((len(soprano_tokens), 4), int(soprano_tokens[0]), dtype=np.int64)
    output[:, 0] = np.asarray(soprano_tokens, dtype=np.int64)
    with torch.no_grad():
        logits = model(soprano)
        for voice_index, voice_logits in enumerate(logits, start=1):
            voice_name = ["alto", "tenor", "bass"][voice_index - 1]
            for t in range(len(soprano_tokens)):
                output[t, voice_index] = int(
                    sample_from_allowed_logits(
                        voice_logits[0, t].detach().cpu(),
                        allowed_ids_by_voice[voice_name],
                        temperature=temperature,
                        top_k=8,
                        banned_ids=banned_ids,
                    ).item()
                )
    return output


def make_custom_soprano_tokens(token_to_id: dict[Any, int], max_steps: int = 96) -> np.ndarray:
    """Return the demo soprano melody for final conditioned generation."""
    custom_soprano_midi_quarter = [
        60,
        62,
        64,
        67,
        69,
        67,
        65,
        64,
        65,
        67,
        69,
        67,
        64,
        62,
        60,
        60,
        67,
        69,
        67,
        65,
        64,
        65,
        67,
        64,
        62,
        64,
        65,
        62,
        60,
        62,
        60,
        60,
    ]

    expanded: list[int] = []
    for pitch in custom_soprano_midi_quarter:
        expanded.extend([pitch, pitch])
    expanded = expanded[:max_steps]
    return np.asarray([int(token_to_id[pitch]) for pitch in expanded], dtype=np.int64)


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Train the conditioned baseline/model and write outputs."""
    ensure_project_dirs(config)
    set_seed(int(config["seed"]))
    dataset = _load_or_build_dataset(config)
    token_to_id = dataset["token_to_id"]
    id_to_token = dataset["id_to_token"]
    pad_id = special_id(token_to_id, "PAD")
    banned_ids = [special_id(token_to_id, name) for name in ["PAD", "REST", "HOLD", "START", "END"]]
    allowed_ids_by_voice = allowed_pitch_ids(token_to_id)
    device = resolve_device(config["training"]["device"])

    steps_per_measure = int(round(4.0 / float(dataset["metadata"]["grid"])))
    baseline = LookupHarmonizationBaseline(id_to_token=id_to_token, pad_id=pad_id, steps_per_measure=steps_per_measure).fit(dataset["train"])
    baseline_score = baseline.score(dataset["test"])

    model_cfg = config["conditioned_model"]
    model = SopranoConditionedHarmonizer(
        vocab_size=int(dataset["metadata"]["vocab_size"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        bidirectional=bool(model_cfg["bidirectional"]),
        pad_id=pad_id,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    batch_size = int(config["training"]["batch_size"])
    history: list[dict[str, float]] = []
    train_sequences = list(dataset["train"])

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        random.shuffle(train_sequences)
        model.train()
        train_losses: list[float] = []
        for batch in batched(train_sequences, batch_size):
            soprano, target = _make_harmonizer_batch(batch, pad_id, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _harmonizer_loss(model(soprano), target, criterion)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_stats = _evaluate_model(model, dataset["val"], pad_id, banned_ids, batch_size, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_stats["loss"]})
        print(f"[conditioned] epoch {epoch:02d} train={history[-1]['train_loss']:.3f} val={val_stats['loss']:.3f}")

    test_stats = _evaluate_model(model, dataset["test"], pad_id, banned_ids, batch_size, device)
    sample_len = min(int(model_cfg["sample_length"]), len(dataset["test"][0]))
    if bool(model_cfg.get("use_custom_demo_melody", False)):
        soprano_tokens = make_custom_soprano_tokens(token_to_id, max_steps=int(model_cfg["sample_length"]))
    else:
        soprano_tokens = dataset["test"][0][:sample_len, 0]
    baseline_tokens = baseline.predict(soprano_tokens)
    model_tokens = harmonize_with_beam_search(
        model,
        soprano_tokens,
        allowed_ids_by_voice=allowed_ids_by_voice,
        id_to_token=id_to_token,
        device=device,
        beam_size=int(model_cfg.get("beam_size", 8)),
        top_k_per_voice=int(model_cfg.get("top_k_per_voice", 5)),
        rule_weight=float(model_cfg.get("rule_weight", 1.0)),
        steps_per_measure=steps_per_measure,
    )
    baseline_matrix = polish_generated_sequence(decode_token_matrix(baseline_tokens, id_to_token), max_repeat_steps=6)
    model_matrix = polish_generated_sequence(decode_token_matrix(model_tokens, id_to_token), max_repeat_steps=6)

    paths = config["paths"]
    midi_dir = PROJECT_ROOT / paths["midi_dir"]
    tables_dir = PROJECT_ROOT / paths["tables_dir"]
    figures_dir = PROJECT_ROOT / paths["figures_dir"]
    checkpoints_dir = PROJECT_ROOT / paths["checkpoints_dir"]
    logs_dir = PROJECT_ROOT / paths["logs_dir"]

    satb_matrix_to_midi(baseline_matrix, midi_dir / "baseline_conditioned.mid", grid=float(dataset["metadata"]["grid"]))
    satb_matrix_to_midi(model_matrix, midi_dir / "symbolic_conditioned.mid", grid=float(dataset["metadata"]["grid"]))

    ref_hist = aggregate_pitch_histogram(dataset["train_pitches"])
    rows = [
        {
            "model": "Lookup baseline",
            **baseline_score,
            **sequence_music_metrics(baseline_matrix, ref_hist, steps_per_measure),
            "notes": "soprano pitch-class plus beat-position lookup",
        },
        {
            "model": "BiGRU model",
            "alto_accuracy": test_stats["alto_accuracy"],
            "tenor_accuracy": test_stats["tenor_accuracy"],
            "bass_accuracy": test_stats["bass_accuracy"],
            "average_voice_accuracy": test_stats["average_voice_accuracy"],
            "exact_atb_accuracy": test_stats["exact_atb_accuracy"],
            **sequence_music_metrics(model_matrix, ref_hist, steps_per_measure),
            "notes": "bidirectional GRU with beam search and SATB rule penalties",
        },
    ]
    save_table(rows, tables_dir / "conditioned_metrics.csv")
    save_json(history, logs_dir / "conditioned_history.json")
    plot_training_curve(history, figures_dir / "conditioned_training_curve.png", "Conditioned BiGRU Training Curve")
    plot_metric_comparison(
        tables_dir / "conditioned_metrics.csv",
        figures_dir / "conditioned_metric_comparison.png",
        ["average_voice_accuracy", "exact_atb_accuracy", "strong_beat_consonance"],
        "Conditioned Harmonization Metrics",
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "test_stats": test_stats,
        },
        checkpoints_dir / "conditioned_bigru.pt",
    )
    return {"test_loss": test_stats["loss"], "midi": str(midi_dir / "symbolic_conditioned.mid")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    run(config)


if __name__ == "__main__":
    main()
