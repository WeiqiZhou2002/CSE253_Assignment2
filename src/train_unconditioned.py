from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .baselines import MarkovSATBBaseline
from .data import build_dataset, load_processed_dataset, save_processed_dataset
from .evaluate import aggregate_pitch_histogram, save_table, sequence_music_metrics
from .midi_io import satb_matrix_to_midi
from .models import SATBGRULanguageModel
from .plots import plot_metric_comparison, plot_training_curve
from .sample import sample_from_logits
from .tokenization import decode_token_matrix, special_id
from .utils import PROJECT_ROOT, batched, ensure_project_dirs, load_config, resolve_device, save_json, set_seed


def _load_or_build_dataset(config: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / config["paths"]["processed_data"]
    if path.exists():
        return load_processed_dataset(path)
    dataset = build_dataset(seed=config["seed"], vocab_config=config["vocab"], **config["data"])
    save_processed_dataset(dataset, path)
    return dataset


def _make_lm_batch(sequences: list[np.ndarray], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) - 1 for seq in sequences)
    x = torch.full((len(sequences), max_len, 4), pad_id, dtype=torch.long)
    y = torch.full((len(sequences), max_len, 4), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        n = len(seq) - 1
        x[i, :n] = torch.as_tensor(seq[:-1], dtype=torch.long)
        y[i, :n] = torch.as_tensor(seq[1:], dtype=torch.long)
    return x.to(device), y.to(device)


def _lm_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    target: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    loss = torch.zeros((), device=target.device)
    for voice_index, logits in enumerate(outputs):
        loss = loss + criterion(logits.reshape(-1, logits.shape[-1]), target[:, :, voice_index].reshape(-1))
    return loss


def _evaluate_loss(
    model: SATBGRULanguageModel,
    sequences: list[np.ndarray],
    pad_id: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    losses: list[float] = []
    with torch.no_grad():
        for batch in batched(sequences, batch_size):
            x, y = _make_lm_batch(batch, pad_id, device)
            losses.append(float(_lm_loss(model(x), y, criterion).detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def generate_unconditioned(
    model: SATBGRULanguageModel,
    start_state: np.ndarray,
    length: int,
    temperature: float,
    banned_ids: list[int],
    device: torch.device,
) -> np.ndarray:
    """Sample a token matrix from an autoregressive model."""
    model.eval()
    generated = [list(map(int, start_state))]
    with torch.no_grad():
        for _ in range(length - 1):
            x = torch.as_tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
            outputs = model(x)
            next_state = [
                int(sample_from_logits(outputs[voice][0, -1].detach().cpu(), temperature, banned_ids).item())
                for voice in range(4)
            ]
            generated.append(next_state)
    return np.asarray(generated, dtype=np.int64)


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Train the unconditioned baseline/model and write outputs."""
    ensure_project_dirs(config)
    set_seed(int(config["seed"]))
    dataset = _load_or_build_dataset(config)
    token_to_id = dataset["token_to_id"]
    id_to_token = dataset["id_to_token"]
    pad_id = special_id(token_to_id, "PAD")
    banned_ids = [special_id(token_to_id, name) for name in ["PAD", "REST", "HOLD", "START", "END"]]
    device = resolve_device(config["training"]["device"])

    markov = MarkovSATBBaseline(pad_id=pad_id, seed=int(config["seed"])).fit(dataset["train"])
    markov_score = markov.score(dataset["test"])
    markov_sample_tokens = markov.sample(length=int(config["unconditioned_model"]["sample_length"]), temperature=1.0)
    markov_sample = decode_token_matrix(markov_sample_tokens, id_to_token)

    model_cfg = config["unconditioned_model"]
    model = SATBGRULanguageModel(
        vocab_size=int(dataset["metadata"]["vocab_size"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
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
            x, y = _make_lm_batch(batch, pad_id, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _lm_loss(model(x), y, criterion)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = _evaluate_loss(model, dataset["val"], pad_id, batch_size, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        print(f"[unconditioned] epoch {epoch:02d} train={history[-1]['train_loss']:.3f} val={val_loss:.3f}")

    test_loss = _evaluate_loss(model, dataset["test"], pad_id, batch_size, device)
    test_perplexity = float(math.exp(min(test_loss / 4.0, 20.0)))
    start_state = markov.sample(length=1)[0]
    sample_tokens = generate_unconditioned(
        model,
        start_state=start_state,
        length=int(model_cfg["sample_length"]),
        temperature=float(model_cfg["temperature"]),
        banned_ids=banned_ids,
        device=device,
    )
    sample_matrix = decode_token_matrix(sample_tokens, id_to_token)

    paths = config["paths"]
    midi_dir = PROJECT_ROOT / paths["midi_dir"]
    tables_dir = PROJECT_ROOT / paths["tables_dir"]
    figures_dir = PROJECT_ROOT / paths["figures_dir"]
    checkpoints_dir = PROJECT_ROOT / paths["checkpoints_dir"]
    logs_dir = PROJECT_ROOT / paths["logs_dir"]

    satb_matrix_to_midi(markov_sample, midi_dir / "baseline_unconditioned.mid", grid=float(dataset["metadata"]["grid"]))
    satb_matrix_to_midi(sample_matrix, midi_dir / "symbolic_unconditioned.mid", grid=float(dataset["metadata"]["grid"]))

    ref_hist = aggregate_pitch_histogram(dataset["train_pitches"])
    rows = [
        {
            "model": "Markov baseline",
            "test_loss": markov_score["nll"],
            "perplexity": markov_score["perplexity"],
            **sequence_music_metrics(markov_sample, ref_hist),
            "notes": "first-order SATB tuple transitions",
        },
        {
            "model": "GRU model",
            "test_loss": test_loss,
            "perplexity": test_perplexity,
            **sequence_music_metrics(sample_matrix, ref_hist),
            "notes": "four-head autoregressive GRU",
        },
    ]
    save_table(rows, tables_dir / "unconditioned_metrics.csv")
    save_json(history, logs_dir / "unconditioned_history.json")
    plot_training_curve(history, figures_dir / "unconditioned_training_curve.png", "Unconditioned GRU Training Curve")
    plot_metric_comparison(
        tables_dir / "unconditioned_metrics.csv",
        figures_dir / "unconditioned_metric_comparison.png",
        ["range_violations", "strong_beat_consonance", "chord_diversity"],
        "Unconditioned Music Metrics",
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
            "test_loss": test_loss,
        },
        checkpoints_dir / "unconditioned_gru.pt",
    )
    return {"test_loss": test_loss, "midi": str(midi_dir / "symbolic_unconditioned.mid")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    run(config)


if __name__ == "__main__":
    main()

