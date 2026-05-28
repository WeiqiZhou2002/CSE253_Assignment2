from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics_music import VOICE_RANGES, pitch_class_histogram
from .tokenization import valid_pitch
from .utils import ensure_dir


plt.switch_backend("Agg")


def plot_sequence_lengths(lengths: list[int], output_path: str | Path) -> Path:
    """Save a histogram of sequence lengths."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=16, color="#4C78A8", edgecolor="white")
    ax.set_title("Chorale Sequence Lengths")
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Number of sequences")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_voice_ranges(dataset_pitches: list[np.ndarray], output_path: str | Path) -> Path:
    """Save pitch distributions by SATB voice."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    voice_names = ["soprano", "alto", "tenor", "bass"]
    data = []
    for voice_index in range(4):
        pitches = []
        for seq in dataset_pitches:
            pitches.extend([int(v) for v in np.asarray(seq)[:, voice_index] if valid_pitch(v)])
        data.append(pitches)
    ax.boxplot(data, labels=[v.title() for v in voice_names], showfliers=False)
    for i, voice in enumerate(voice_names, start=1):
        low, high = VOICE_RANGES[voice]
        ax.plot([i - 0.32, i + 0.32], [low, low], color="#E45756", linewidth=1)
        ax.plot([i - 0.32, i + 0.32], [high, high], color="#E45756", linewidth=1)
    ax.set_title("Pitch Ranges by Voice")
    ax.set_ylabel("MIDI pitch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_pitch_class_histogram(dataset_pitches: list[np.ndarray], output_path: str | Path) -> Path:
    """Save an aggregate pitch-class histogram."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    hist = np.zeros(12, dtype=np.float64)
    for seq in dataset_pitches:
        hist += pitch_class_histogram(seq)
    hist = hist / hist.sum() if hist.sum() else hist
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]
    ax.bar(labels, hist, color="#59A14F")
    ax.set_title("Aggregate Pitch-Class Distribution")
    ax.set_ylabel("Normalized frequency")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_training_curve(history: list[dict[str, Any]], output_path: str | Path, title: str) -> Path:
    """Save a train/validation loss curve."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["val_loss"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train, marker="o", label="train")
    ax.plot(epochs, val, marker="o", label="validation")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_metric_comparison(csv_path: str | Path, output_path: str | Path, columns: list[str], title: str) -> Path:
    """Save a compact grouped-bar comparison plot from a metrics CSV."""
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    df = pd.read_csv(csv_path)
    available = [col for col in columns if col in df.columns]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    width = 0.8 / max(1, len(available))
    for i, col in enumerate(available):
        ax.bar(x + i * width, df[col], width=width, label=col)
    ax.set_xticks(x + width * max(0, len(available) - 1) / 2)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path

