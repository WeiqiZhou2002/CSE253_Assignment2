from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path
from typing import Any

import nbformat as nbf
import pandas as pd
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_processed_dataset
from src.utils import ensure_dir, load_config


def _data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _image(path: Path, alt: str) -> str:
    if not path.exists():
        return f"*Missing figure: `{path.name}`*"
    return f'<img alt="{alt}" src="{_data_url(path)}" style="max-width: 100%; height: auto;">'


def _table(path: Path) -> str:
    if not path.exists():
        return f"*Missing table: `{path.name}`*"
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(3)
    return df.to_html(index=False, border=0)


def _snippet(path: Path, start: str, end: str | None = None, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start_idx = next(i for i, line in enumerate(lines) if start in line)
    except StopIteration:
        return ""
    end_idx = len(lines)
    if end:
        for i in range(start_idx + 1, len(lines)):
            if end in lines[i]:
                end_idx = i
                break
    snippet = "\n".join(lines[start_idx : min(end_idx, start_idx + max_lines)])
    return f"```python\n{snippet}\n```"


def build_notebook(config: dict[str, Any]) -> nbf.NotebookNode:
    dataset = load_processed_dataset(ROOT / config["paths"]["processed_data"])
    meta = dataset["metadata"]
    figures_dir = ROOT / config["paths"]["figures_dir"]
    tables_dir = ROOT / config["paths"]["tables_dir"]
    midi_dir = ROOT / config["paths"]["midi_dir"]
    midi_rel = Path(config["paths"]["midi_dir"])

    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            "# BachBot: Unconditioned Chorale Generation and Melody-Conditioned Harmonization\n\n"
            "This workbook documents two symbolic music-generation tasks for the assignment: "
            "full SATB chorale generation and soprano-conditioned harmonization. "
            "It is written to be readable after HTML export, so peer graders do not need to run the code."
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Assignment Tasks Attempted\n\n"
            "1. **Symbolic, unconditioned generation**: an autoregressive GRU learns to generate complete SATB time steps.\n"
            "2. **Symbolic, conditioned generation**: a bidirectional GRU harmonizes a given soprano line with alto, tenor, and bass.\n\n"
            "Final generated files:\n\n"
            f"- `{midi_rel / 'symbolic_unconditioned.mid'}`\n"
            f"- `{midi_rel / 'symbolic_conditioned.mid'}`"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Dataset and Preprocessing\n\n"
            f"Dataset source used in this run: **{meta['source']}**.\n\n"
            f"{meta['source_note']}\n\n"
            "Each piece is represented as a fixed-grid matrix with shape `T x 4`, ordered as "
            "`[soprano, alto, tenor, bass]`. Values are MIDI pitch tokens plus reserved symbols for "
            "`PAD`, `REST`, `HOLD`, `START`, and `END`. The grid size is "
            f"{meta['grid']} quarter lengths per step."
        )
    )
    cells.append(nbf.v4.new_markdown_cell("### Split Summary\n\n" + _table(tables_dir / "dataset_summary.csv")))
    cells.append(nbf.v4.new_markdown_cell("### Sequence Lengths\n\n" + _image(figures_dir / "sequence_lengths.png", "Sequence length histogram")))
    cells.append(nbf.v4.new_markdown_cell("### Voice Ranges\n\n" + _image(figures_dir / "voice_ranges.png", "Voice range boxplots")))
    cells.append(nbf.v4.new_markdown_cell("### Pitch-Class Distribution\n\n" + _image(figures_dir / "pitch_class_histogram.png", "Pitch-class histogram")))
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Tokenization and MIDI Output\n\n"
            "The tokenization layer keeps the model vocabulary compact and reversible. MIDI export collapses repeated "
            "grid values into longer notes and writes separate tracks for the four voices.\n\n"
            + _snippet(ROOT / "src" / "tokenization.py", "def build_vocab", "def token_id", max_lines=40)
            + "\n\n"
            + _snippet(ROOT / "src" / "midi_io.py", "def satb_matrix_to_midi", "def write_reference_midi", max_lines=70)
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Task 1: Symbolic Unconditioned Generation\n\n"
            "The unconditioned task is modeled as next-step prediction over SATB tuples. "
            "The baseline is a first-order Markov model over complete four-voice states. "
            "The neural model embeds each voice, concatenates the embeddings, processes the sequence with a GRU, "
            "and predicts the next soprano, alto, tenor, and bass tokens with separate output heads.\n\n"
            + _snippet(ROOT / "src" / "models.py", "class SATBGRULanguageModel", "class SopranoConditionedHarmonizer", max_lines=70)
        )
    )
    cells.append(nbf.v4.new_markdown_cell("### Task 1 Training Curve\n\n" + _image(figures_dir / "unconditioned_training_curve.png", "Unconditioned training curve")))
    cells.append(nbf.v4.new_markdown_cell("### Task 1 Evaluation\n\n" + _table(tables_dir / "unconditioned_metrics.csv")))
    cells.append(nbf.v4.new_markdown_cell("### Task 1 Metric Comparison\n\n" + _image(figures_dir / "unconditioned_metric_comparison.png", "Unconditioned metric comparison")))
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Task 2: Soprano-Conditioned Harmonization\n\n"
            "For conditioned generation, the full soprano melody is known in advance. "
            "The baseline predicts the most common lower-voice chord for a soprano pitch class and beat position, "
            "with backoff to pitch-class-only and global counts. The neural model uses a bidirectional GRU over the "
            "soprano sequence and predicts alto, tenor, and bass at every time step.\n\n"
            + _snippet(ROOT / "src" / "baselines.py", "class LookupHarmonizationBaseline", max_lines=75)
            + "\n\n"
            + _snippet(ROOT / "src" / "models.py", "class SopranoConditionedHarmonizer", max_lines=65)
        )
    )
    cells.append(nbf.v4.new_markdown_cell("### Task 2 Training Curve\n\n" + _image(figures_dir / "conditioned_training_curve.png", "Conditioned training curve")))
    cells.append(nbf.v4.new_markdown_cell("### Task 2 Evaluation\n\n" + _table(tables_dir / "conditioned_metrics.csv")))
    cells.append(nbf.v4.new_markdown_cell("### Task 2 Metric Comparison\n\n" + _image(figures_dir / "conditioned_metric_comparison.png", "Conditioned metric comparison")))
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Evaluation Design\n\n"
            "The ML metrics measure predictive fit: validation/test loss for both neural models, perplexity for the "
            "unconditioned model, and per-voice plus exact lower-voice accuracy for harmonization. "
            "Because musical quality is not captured by likelihood alone, the project also reports voice range "
            "violations, voice crossings, consonance on strong beats, repeated-note ratio, chord diversity, "
            "parallel perfect fifths/octaves, pitch-class histogram distance, and a simple cadence heuristic.\n\n"
            + _snippet(ROOT / "src" / "metrics_music.py", "def voice_range_violation_rate", "def cadence_quality_heuristic", max_lines=120)
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Related Work\n\n"
            "**DeepBach** is directly relevant because it models Bach chorales and supports generation and harmonization. "
            "Our implementation is intentionally smaller: a compact grid representation, lightweight GRU models, "
            "and a classroom-scale training pipeline.\n\n"
            "**Music Transformer** is a broader symbolic music-generation reference point. Its self-attention mechanism "
            "can model longer-range structure than a small GRU, but it is more expensive and less necessary for this "
            "minimum viable chorale pipeline.\n\n"
            "Bach chorales are a common benchmark because they are compact, symbolic, four-voice, harmonically regular, "
            "and interpretable using both ML metrics and music-theory-inspired heuristics."
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Limitations and Future Work\n\n"
            "- The eighth-note grid is simple and loses some expressive rhythmic detail.\n"
            "- The neural models are deliberately small; longer training and attention-based models could improve global form.\n"
            "- Voice-leading metrics are heuristics, not substitutes for human listening.\n"
            "- If `music21` is installed, the same pipeline can train on the real Bach chorale corpus instead of the fallback data.\n\n"
            "Useful extensions include rule-based postprocessing, temperature comparisons, custom melody harmonization, "
            "and optional MIDI-to-MP3 rendering for easier presentation playback."
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Reproducibility\n\n"
            "Run the full pipeline with:\n\n"
            "```bash\n"
            "python3 scripts/run_all.py --config configs/mvp.yaml\n"
            "```\n\n"
            "The command rebuilds the dataset, EDA figures, model checkpoints, metric tables, MIDI files, notebook HTML, "
            "and the `submission/` folder."
        )
    )

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    return nb


def build_and_export(config: dict[str, Any]) -> tuple[Path, Path]:
    notebooks_dir = ROOT / "notebooks"
    ensure_dir(notebooks_dir)
    nb = build_notebook(config)
    ipynb_path = notebooks_dir / "workbook.ipynb"
    html_path = notebooks_dir / "workbook.html"
    nbf.write(nb, ipynb_path)

    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    body = body.lstrip()
    if not body.startswith("<!DOCTYPE html>"):
        body = "<!DOCTYPE html>\n" + body
    html_path.write_text(body, encoding="utf-8")
    return ipynb_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    build_and_export(config)
    print("Exported notebooks/workbook.ipynb and notebooks/workbook.html")


if __name__ == "__main__":
    main()
