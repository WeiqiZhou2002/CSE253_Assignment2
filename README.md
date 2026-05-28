# BachBot: Symbolic Chorale Generation and Harmonization

This repository implements the project described in `bachbot_group_project_plan.md` for two CSE 153/253 Assignment 2 tasks:

1. **Symbolic, unconditioned generation**: train a model that generates complete SATB chorales.
2. **Symbolic, conditioned generation**: train a model that harmonizes a soprano melody with alto, tenor, and bass.

The code prefers the `music21` Bach chorale corpus when `music21` is installed. If it is unavailable, it falls back to a deterministic chorale-style SATB corpus so the full pipeline remains runnable in a clean local environment.

## Quick Start

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_all.py --config configs/mvp.yaml
```

The main outputs are:

```text
outputs/midi/symbolic_unconditioned.mid
outputs/midi/symbolic_conditioned.mid
outputs/tables/*.csv
outputs/figures/*.png
notebooks/workbook.ipynb
notebooks/workbook.html
submission/workbook.html
submission/symbolic_unconditioned.mid
submission/symbolic_conditioned.mid
```

`submission/video_url.txt` is created as a placeholder because the final presentation video must be recorded and uploaded by the group.

## Project Layout

```text
configs/              YAML configuration
src/                  data, models, baselines, metrics, plotting, MIDI I/O
scripts/              end-to-end runner, notebook export, packaging
notebooks/            generated notebook and HTML export
outputs/              processed data, figures, tables, MIDI, checkpoints
submission/           files with names expected by the autograder
```

## Useful Commands

Run everything:

```bash
python3 scripts/run_all.py --config configs/mvp.yaml
```

Train only the unconditioned model:

```bash
python3 -m src.train_unconditioned --config configs/mvp.yaml
```

Train only the conditioned harmonizer:

```bash
python3 -m src.train_conditioned --config configs/mvp.yaml
```

Rebuild the notebook from existing outputs:

```bash
python3 scripts/export_notebook.py --config configs/mvp.yaml
```

Rebuild the submission folder:

```bash
python3 scripts/make_submission.py --config configs/mvp.yaml
```

