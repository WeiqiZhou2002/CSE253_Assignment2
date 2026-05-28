# BachBot Group Project Plan

Project title: **BachBot: Unconditioned Chorale Generation and Melody-Conditioned Harmonization**

Group size: **3 members**

Primary goal: Build a complete, reproducible music-generation pipeline for two symbolic-generation tasks, then present it clearly in a clean notebook and a 20-minute video.

Recommended scope: Use Bach chorales as the dataset, train your own lightweight models, generate two MIDI files, and evaluate both ML performance and musical quality.

---

## 1. Assignment Fit and Required Submission Files

This plan targets two assignment options:

1. **Symbolic, unconditioned generation**
   - Train a model that learns a distribution over complete SATB chorale sequences.
   - Generate a new 4-voice Bach-style chorale from scratch.
   - Output file: `symbolic_unconditioned.mid`

2. **Symbolic, conditioned generation**
   - Given a soprano melody, generate alto, tenor, and bass parts.
   - Output file: `symbolic_conditioned.mid`

Required final files:

```text
workbook.html
video_url.txt
symbolic_unconditioned.mid
symbolic_conditioned.mid
```

Important constraints:

- The notebook should be exported to HTML and should be readable without requiring peer graders to run the code.
- The presentation should be around 20 minutes.
- The project should train its own model weights on a training set.
- The presentation and notebook should explicitly cover, for each task:
  1. Exploratory analysis, data collection, preprocessing, and discussion
  2. Modeling
  3. Evaluation
  4. Related work

---

## 2. Project Story

One strong framing for the final presentation:

> We trained two symbolic music-generation systems on Bach chorales. The first generates complete 4-part chorales from scratch. The second harmonizes a given soprano melody by generating alto, tenor, and bass parts. We compare simple statistical baselines against neural sequence models and evaluate both predictive performance and music-theory-inspired quality metrics.

This story is simple, coherent, and maps directly to the grading rubric.

---

## 3. Technical Approach Summary

### Dataset

Use Bach chorales from `music21`.

Suggested loading route:

```python
from music21 import corpus
chorales = list(corpus.chorales.Iterator())
```

Fallback routes if the exact API differs:

```python
from music21 import corpus
scores = corpus.getComposer('bach')
```

or use a small set of local MIDI/XML files if `music21` corpus access is difficult.

### Representation

Represent each chorale as a time-indexed matrix:

```text
shape = (T, 4)
columns = [soprano, alto, tenor, bass]
values = MIDI pitch tokens plus special tokens
```

Special tokens:

```text
PAD   = padding for batching
REST  = silence
HOLD  = continuation from previous note, optional
START = sequence start, optional
END   = sequence end, optional
```

For an MVP, avoid overcomplicating rhythm. Quantize to a fixed grid:

```text
quarter note grid: 1.0 quarterLength
or eighth note grid: 0.5 quarterLength
```

Recommended MVP: **eighth-note grid**, because it captures more movement while staying manageable.

### Voice order

Always use the same voice order:

```text
0 = soprano
1 = alto
2 = tenor
3 = bass
```

### Suggested pitch ranges for validation metrics

These are approximate and should be configurable:

```python
VOICE_RANGES = {
    'soprano': (60, 81),  # C4 to A5
    'alto':    (55, 74),  # G3 to D5
    'tenor':   (48, 67),  # C3 to G4
    'bass':    (40, 60),  # E2 to C4
}
```

---

## 4. Task 1: Symbolic Unconditioned Generation

### 4.1 Problem Formulation

Train an autoregressive model that predicts the next SATB time step from previous time steps.

Input:

```text
x_0, x_1, ..., x_{t-1}
```

Target:

```text
x_t = [soprano_t, alto_t, tenor_t, bass_t]
```

The model learns:

```text
p(x_t | x_<t)
```

Generation starts from a `START` token or an initial seed and samples time steps until the desired length or an `END` token.

### 4.2 Baseline Model

Implement a Markov baseline over SATB chord tuples.

Basic version:

```text
state_t = tuple(S, A, T, B) at time t
predict state_t from state_{t-1}
```

Fallback if too sparse:

```text
predict each voice independently from its previous note
```

Useful baseline outputs:

- `outputs/baseline_unconditioned.mid`
- Baseline perplexity or negative log likelihood if implemented probabilistically
- Musical metrics table

### 4.3 Neural Model

Recommended model: small GRU language model with four output heads.

High-level architecture:

```text
SATB token input
-> per-voice embeddings
-> concatenate voice embeddings
-> GRU / LSTM
-> four linear output heads
   -> soprano logits
   -> alto logits
   -> tenor logits
   -> bass logits
```

Loss:

```python
loss = CE(soprano_logits, soprano_target)
     + CE(alto_logits, alto_target)
     + CE(tenor_logits, tenor_target)
     + CE(bass_logits, bass_target)
```

Recommended MVP hyperparameters:

```yaml
embedding_dim: 64
hidden_dim: 128
num_layers: 2
dropout: 0.2
batch_size: 16
learning_rate: 0.001
epochs: 20
max_seq_len: 256
quantization: 0.5
```

### 4.4 Sampling

Implement temperature sampling:

```python
def sample_from_logits(logits, temperature=1.0):
    logits = logits / temperature
    probs = softmax(logits)
    return categorical_sample(probs)
```

Generate at least three samples:

```text
temperature = 0.7  # safer, more conservative
temperature = 1.0  # normal
temperature = 1.3  # more varied, possibly chaotic
```

Pick the best-sounding one for final submission as:

```text
symbolic_unconditioned.mid
```

---

## 5. Task 2: Symbolic Conditioned Generation

### 5.1 Problem Formulation

Given a soprano melody, generate alto, tenor, and bass.

Input:

```text
soprano_0, soprano_1, ..., soprano_T
```

Target:

```text
[alto_t, tenor_t, bass_t] for each time step t
```

The model learns:

```text
p(alto_t, tenor_t, bass_t | soprano_0:T)
```

This is a harmonization task.

### 5.2 Baseline Model

Implement a simple lookup-table baseline:

```text
Given soprano pitch class and beat position,
predict the most common [alto, tenor, bass] tuple from training data.
```

Key conditioning features:

```text
soprano pitch class
soprano absolute pitch bucket
beat strength or time-step modulo measure
previous lower-voice chord, optional
```

If no exact match exists, back off to:

```text
most common lower-voice chord for soprano pitch class
then most common lower-voice chord overall
```

### 5.3 Neural Model

Recommended model: bidirectional GRU harmonizer.

High-level architecture:

```text
soprano token sequence
+ optional beat-position features
-> embedding layer
-> bidirectional GRU
-> three output heads
   -> alto logits
   -> tenor logits
   -> bass logits
```

Loss:

```python
loss = CE(alto_logits, alto_target)
     + CE(tenor_logits, tenor_target)
     + CE(bass_logits, bass_target)
```

Why bidirectional is okay here:

- For harmonization, the full melody is known in advance.
- Looking ahead can produce smoother harmonies.
- This is still conditioned generation because the lower voices are generated from the soprano input.

Recommended MVP hyperparameters:

```yaml
embedding_dim: 64
hidden_dim: 128
num_layers: 2
bidirectional: true
dropout: 0.2
batch_size: 16
learning_rate: 0.001
epochs: 20
max_seq_len: 256
quantization: 0.5
```

### 5.4 Inference

Use either:

1. a soprano melody from the held-out test set, or
2. a short melody written by the group.

Recommended final demo:

- Use a held-out Bach soprano melody first, because ground truth exists for evaluation.
- Optionally also harmonize a short custom melody for the presentation.

Final output:

```text
symbolic_conditioned.mid
```

---

## 6. Evaluation Plan

Do not rely only on subjective listening. Use a mix of ML metrics and music metrics.

### 6.1 ML Metrics

For both tasks:

```text
validation loss
test loss
perplexity
```

For conditioned harmonization:

```text
alto pitch accuracy
tenor pitch accuracy
bass pitch accuracy
average voice accuracy
all-three-voices exact-match accuracy
```

Note: exact-match accuracy may be low because many harmonizations can be valid.

### 6.2 Music-Theory-Inspired Metrics

Implement these for generated samples and, when possible, compare against real test chorales.

| Metric | Why it matters |
|---|---|
| Voice range violation rate | Checks whether voices stay in plausible human singing ranges. |
| Voice crossing count | Checks whether alto goes above soprano, tenor above alto, etc. |
| Consonance rate on strong beats | Checks whether stable beats contain stable intervals. |
| Repeated-note ratio | Detects if the model gets stuck. |
| Pitch-class histogram distance | Compares generated pitch distribution to training data. |
| Chord diversity | Checks whether the model uses varied harmonies. |
| Parallel fifth/octave count | Captures a basic classical harmony issue. |
| Cadence quality heuristic | Checks whether the ending sounds final. |

### 6.3 Baseline Comparison Table

Create one table for each task.

Task 1 table example:

| Model | Test loss | Perplexity | Range violations | Voice crossings | Consonance rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| Random baseline | TBD | TBD | TBD | TBD | TBD | sanity check |
| Markov baseline | TBD | TBD | TBD | TBD | TBD | local transitions |
| GRU model | TBD | TBD | TBD | TBD | TBD | main model |

Task 2 table example:

| Model | Alto acc. | Tenor acc. | Bass acc. | Exact ATB acc. | Range violations | Voice crossings | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Lookup baseline | TBD | TBD | TBD | TBD | TBD | TBD | common chord lookup |
| BiGRU model | TBD | TBD | TBD | TBD | TBD | TBD | main model |

---

## 7. Related Work to Discuss

Use these as conceptual anchors in the notebook and presentation.

### DeepBach

Relevant because it directly models Bach chorales and supports chorale generation/harmonization. Discuss how your project is simpler:

- smaller model
- simpler tokenization
- fewer constraints
- trained as a class project rather than a full research system

### Music Transformer

Relevant as a broader symbolic music generation model. Discuss:

- self-attention for longer-range structure
- why a GRU is more feasible for this project
- what a Transformer might improve if there were more time

### General Bach chorale harmonization work

Discuss why Bach chorales are a standard benchmark:

- compact symbolic dataset
- strong harmonic regularities
- clear SATB structure
- interpretable evaluation using music-theory heuristics

---

## 8. Repository Layout for Codex

Suggested structure:

```text
bachbot/
  README.md
  requirements.txt
  configs/
    mvp.yaml
  notebooks/
    workbook.ipynb
  src/
    __init__.py
    data.py
    tokenization.py
    midi_io.py
    models.py
    train_unconditioned.py
    train_conditioned.py
    baselines.py
    evaluate.py
    metrics_music.py
    sample.py
    plots.py
    utils.py
  scripts/
    run_all.py
    export_notebook.py
    make_submission.py
  outputs/
    figures/
    tables/
    midi/
    checkpoints/
    logs/
  submission/
    workbook.html
    video_url.txt
    symbolic_unconditioned.mid
    symbolic_conditioned.mid
```

---

## 9. Implementation Details for Codex

### 9.1 `src/data.py`

Responsibilities:

- Load Bach chorales.
- Extract four SATB voices.
- Quantize to fixed time grid.
- Convert each chorale to a `(T, 4)` integer/pitch matrix.
- Split into train, validation, and test sets.
- Save processed dataset to disk.

Suggested functions:

```python
def load_bach_chorales(max_chorales=None):
    """Return a list of music21 scores."""


def score_to_satb_matrix(score, grid=0.5):
    """Convert one score into a T x 4 matrix of MIDI pitches/special tokens."""


def build_dataset(grid=0.5, max_seq_len=256):
    """Return train/val/test datasets plus metadata."""


def save_processed_dataset(dataset, path):
    """Save processed arrays and metadata."""


def load_processed_dataset(path):
    """Load processed arrays and metadata."""
```

Data split recommendation:

```text
70% train
15% validation
15% test
```

Use a fixed random seed.

### 9.2 `src/tokenization.py`

Responsibilities:

- Map MIDI pitches and special symbols to integer IDs.
- Convert SATB pitch matrices to token matrices.
- Convert token matrices back to pitches.

Suggested token set:

```python
SPECIAL_TOKENS = ['PAD', 'REST', 'HOLD', 'START', 'END']
PITCHES = list(range(21, 109))
```

Suggested functions:

```python
def build_vocab():
    """Return token_to_id and id_to_token."""


def encode_pitch_matrix(matrix, token_to_id):
    """Convert pitch/rest/hold values to token ids."""


def decode_token_matrix(tokens, id_to_token):
    """Convert token ids back to MIDI pitches/special values."""
```

### 9.3 `src/models.py`

Models to implement:

```python
class SATBGRULanguageModel(nn.Module):
    """Autoregressive GRU for unconditioned SATB generation."""


class SopranoConditionedHarmonizer(nn.Module):
    """Bidirectional GRU that predicts alto, tenor, and bass from soprano."""
```

`SATBGRULanguageModel.forward(x)`:

```text
x shape: [batch, time, 4]
returns logits for each voice:
  soprano_logits: [batch, time, vocab_size]
  alto_logits:    [batch, time, vocab_size]
  tenor_logits:   [batch, time, vocab_size]
  bass_logits:    [batch, time, vocab_size]
```

`SopranoConditionedHarmonizer.forward(soprano)`:

```text
soprano shape: [batch, time]
returns logits for lower voices:
  alto_logits:  [batch, time, vocab_size]
  tenor_logits: [batch, time, vocab_size]
  bass_logits:  [batch, time, vocab_size]
```

### 9.4 `src/baselines.py`

Implement:

```python
class MarkovSATBBaseline:
    def fit(self, sequences): ...
    def sample(self, length, temperature=1.0): ...
    def score(self, sequences): ...


class LookupHarmonizationBaseline:
    def fit(self, sequences): ...
    def predict(self, soprano_sequence): ...
```

### 9.5 `src/train_unconditioned.py`

Responsibilities:

- Load processed dataset.
- Train Markov baseline.
- Train GRU language model.
- Save checkpoint.
- Save training curves.
- Generate MIDI sample.

Command:

```bash
python -m src.train_unconditioned --config configs/mvp.yaml
```

Expected outputs:

```text
outputs/checkpoints/unconditioned_gru.pt
outputs/figures/unconditioned_training_curve.png
outputs/midi/symbolic_unconditioned.mid
outputs/tables/unconditioned_metrics.csv
```

### 9.6 `src/train_conditioned.py`

Responsibilities:

- Load processed dataset.
- Train lookup harmonization baseline.
- Train BiGRU harmonizer.
- Save checkpoint.
- Save training curves.
- Generate MIDI harmonization sample.

Command:

```bash
python -m src.train_conditioned --config configs/mvp.yaml
```

Expected outputs:

```text
outputs/checkpoints/conditioned_bigru.pt
outputs/figures/conditioned_training_curve.png
outputs/midi/symbolic_conditioned.mid
outputs/tables/conditioned_metrics.csv
```

### 9.7 `src/metrics_music.py`

Suggested functions:

```python
def voice_range_violation_rate(sequence, voice_ranges):
    """Return fraction of notes outside expected range."""


def voice_crossing_count(sequence):
    """Count time steps where S < A, A < T, or T < B."""


def consonance_rate(sequence, strong_beat_mask=None):
    """Estimate consonant interval rate on all or strong beats."""


def repeated_note_ratio(sequence):
    """Fraction of adjacent notes that repeat within each voice."""


def chord_diversity(sequence):
    """Number or ratio of unique SATB sonorities."""


def parallel_fifths_octaves(sequence):
    """Heuristic count of parallel perfect fifths and octaves."""


def pitch_class_histogram(sequence):
    """Return normalized pitch-class histogram."""


def histogram_distance(hist_a, hist_b):
    """Return L1 or JS distance between pitch-class histograms."""
```

### 9.8 `src/midi_io.py`

Responsibilities:

- Convert token/pitch matrix to MIDI.
- Use separate instruments/tracks for S, A, T, B.
- Save generated MIDI files.

Suggested function:

```python
def satb_matrix_to_midi(sequence, output_path, grid=0.5, tempo=90):
    """Write a T x 4 SATB sequence to a MIDI file."""
```

Use either `pretty_midi` or `music21` for output.

### 9.9 `src/plots.py`

Generate notebook-ready figures:

```python
def plot_sequence_lengths(lengths, output_path): ...
def plot_voice_ranges(dataset, output_path): ...
def plot_pitch_class_histogram(dataset, output_path): ...
def plot_training_curve(history, output_path): ...
def plot_metric_comparison(metrics_df, output_path): ...
```

---

## 10. Notebook Plan: `notebooks/workbook.ipynb`

The notebook should be written for peer graders. It should not look like scratch work.

Recommended sections:

```text
1. Title and project overview
2. Assignment tasks attempted
3. Dataset: Bach chorales
4. Preprocessing and tokenization
5. Exploratory analysis
6. Task 1: unconditioned generation
   6.1 Problem formulation
   6.2 Baseline model
   6.3 Neural model
   6.4 Training results
   6.5 Generated sample
   6.6 Evaluation
   6.7 Discussion
7. Task 2: conditioned harmonization
   7.1 Problem formulation
   7.2 Baseline model
   7.3 Neural model
   7.4 Training results
   7.5 Generated sample
   7.6 Evaluation
   7.7 Discussion
8. Related work
9. Limitations and future work
10. Final generated music files
11. Reproducibility instructions
```

The notebook should include:

- plots of dataset statistics
- training curves
- metric tables
- short code walkthroughs
- short explanations of design choices
- links or references to the final MIDI files

Do not include huge raw outputs or messy debugging logs.

---

## 11. Presentation Plan for Three Members

Total target: **20 minutes**, not including optional music playback at the end.

### Slide Outline

| Time | Speaker | Section | Content |
|---:|---|---|---|
| 1 min | Member A | Intro | Project goal and two tasks |
| 3 min | Member A | Data and EDA | Bach chorales, SATB representation, plots |
| 4 min | Member A | Task 1 modeling | Markov baseline, GRU model, sampling |
| 4 min | Member B | Task 2 modeling | Harmonization baseline, BiGRU model |
| 3 min | Member C | Evaluation | ML metrics and music-theory metrics for both tasks |
| 2 min | Member C | Results | Tables, generated examples, what worked/failed |
| 2 min | Member B/C | Related work and limitations | DeepBach, Music Transformer, project tradeoffs |
| 1 min | All | Wrap-up | Key takeaways and file outputs |

Optional after the 20-minute content:

```text
Play symbolic_unconditioned.mid
Play symbolic_conditioned.mid
```

### Speaker Responsibilities

#### Member A: Data + Task 1 Lead

Primary responsibilities:

- Bach chorale data loading
- tokenization and preprocessing
- exploratory data analysis
- unconditioned Markov baseline
- unconditioned GRU model
- Task 1 presentation section

Deliverables:

```text
src/data.py
src/tokenization.py
src/train_unconditioned.py
outputs/midi/symbolic_unconditioned.mid
EDA notebook cells
Task 1 slides
```

#### Member B: Task 2 Lead

Primary responsibilities:

- conditioned harmonization formulation
- lookup-table harmonization baseline
- BiGRU harmonizer
- conditioned generation inference
- Task 2 presentation section

Deliverables:

```text
src/train_conditioned.py
src/baselines.py for harmonization
outputs/midi/symbolic_conditioned.mid
Task 2 notebook cells
Task 2 slides
```

#### Member C: Evaluation + Integration Lead

Primary responsibilities:

- music-theory metrics
- metric tables and plots
- related work summary
- notebook cleanup
- HTML export
- video recording logistics
- final submission folder

Deliverables:

```text
src/metrics_music.py
src/evaluate.py
outputs/tables/*.csv
outputs/figures/*.png
notebooks/workbook.ipynb final polish
submission/workbook.html
submission/video_url.txt
```

### Backup Responsibilities

Each major area should have a backup reviewer:

| Area | Owner | Backup |
|---|---|---|
| Data pipeline | Member A | Member C |
| Task 1 model | Member A | Member B |
| Task 2 model | Member B | Member A |
| Evaluation metrics | Member C | Member B |
| Notebook polish | Member C | Member A |
| Presentation/video | Member C | All |

---

## 12. Milestone Plan

Use relative deadlines based on the final due date.

### T-10 to T-8 days: MVP data and baseline

Must finish:

- load Bach chorales
- create SATB matrices
- generate simple MIDI from a real chorale
- train/test split
- Markov baseline for Task 1
- lookup baseline for Task 2

Exit criteria:

```text
A real chorale can be loaded, encoded, decoded, and saved as MIDI.
Both baselines produce MIDI outputs.
```

### T-7 to T-5 days: Neural models

Must finish:

- train unconditioned GRU
- train conditioned BiGRU
- save checkpoints
- save training curves
- produce first generated MIDI files

Exit criteria:

```text
Both neural models train without crashing.
Both models produce MIDI files.
```

### T-4 to T-3 days: Evaluation and figures

Must finish:

- ML metrics
- music metrics
- baseline-vs-neural comparison tables
- plots for EDA and training curves

Exit criteria:

```text
Notebook has all major tables and plots.
```

### T-2 days: Notebook and presentation

Must finish:

- clean notebook
- final narrative
- slide/video outline
- speaker notes
- generated music selected

Exit criteria:

```text
Notebook can be exported to workbook.html.
Presentation can be delivered in about 20 minutes.
```

### T-1 day: Final packaging

Must finish:

- export `workbook.html`
- record/upload presentation video
- create `video_url.txt`
- copy final MIDI files
- run submission sanity checks

Exit criteria:

```text
submission/
  workbook.html
  video_url.txt
  symbolic_unconditioned.mid
  symbolic_conditioned.mid
```

---

## 13. Minimum Viable Product Checklist

The project is minimally complete when all boxes below are checked.

### Data

- [ ] Bach chorales loaded successfully
- [ ] SATB voice extraction works
- [ ] fixed-grid quantization works
- [ ] train/validation/test split exists
- [ ] at least three EDA plots exist

### Task 1

- [ ] Markov or n-gram baseline implemented
- [ ] GRU/LSTM unconditioned model implemented
- [ ] model trains on training data
- [ ] validation loss curve saved
- [ ] generated MIDI saved as `symbolic_unconditioned.mid`
- [ ] task evaluated with at least one ML metric and two music metrics

### Task 2

- [ ] lookup harmonization baseline implemented
- [ ] conditional GRU/LSTM harmonizer implemented
- [ ] model trains on training data
- [ ] validation loss curve saved
- [ ] generated MIDI saved as `symbolic_conditioned.mid`
- [ ] task evaluated with at least one ML metric and two music metrics

### Submission

- [ ] notebook is clean and documented
- [ ] notebook exported to `workbook.html`
- [ ] video recorded and uploaded
- [ ] `video_url.txt` contains one valid link
- [ ] final MIDI files are present

---

## 14. Stretch Goals

Only attempt these after the MVP is done.

### Stretch Goal 1: Temperature Comparison

Generate multiple unconditioned samples:

```text
temperature 0.7
temperature 1.0
temperature 1.3
```

Compare musical metrics and listening impressions.

### Stretch Goal 2: Rule-Based Postprocessing

Improve outputs by applying simple corrections:

- clip notes to voice ranges
- avoid voice crossing
- force final chord to a stable sonority
- reduce very large melodic leaps

### Stretch Goal 3: Custom Melody Harmonization

Create a short melody manually and let the conditioned model harmonize it.

This is good for the final presentation because it makes the result feel more interactive.

### Stretch Goal 4: MIDI-to-MP3 Rendering

Render the MIDI to MP3 for easier playback in the presentation.

Keep the final required symbolic `.mid` files even if MP3s are also created.

---

## 15. Risks and Fallbacks

### Risk 1: Full SATB chord vocabulary is too sparse

Fallback:

- do not model SATB chords as one huge token
- use four separate voice output heads

### Risk 2: `music21` voice extraction is messy

Fallback:

- filter to chorales that clearly have four parts
- use only a subset of clean chorales
- document the filtering in the notebook

### Risk 3: Neural model sounds worse than baseline

Fallback:

- report honestly
- explain that the baseline captures local harmonic regularities well
- emphasize that the neural model may need more data, better representation, or longer training
- still compare metrics and show generated samples

### Risk 4: Generated output has many voice crossings

Fallback:

- add postprocessing to enforce voice order
- or report voice crossings as a limitation

### Risk 5: Not enough time

Fallback plan:

1. Finish data pipeline.
2. Finish both baselines.
3. Train simple neural models for a few epochs.
4. Generate MIDI.
5. Focus the presentation on clear evaluation and limitations.

A simple but complete pipeline is better than an ambitious incomplete one.

---

## 16. Suggested Codex Prompts

Use these prompts when asking Codex to implement pieces of the project.

### Prompt 1: Data pipeline

```text
Implement src/data.py and src/tokenization.py for a Bach chorale project. Load chorales using music21, extract SATB parts, quantize to a fixed grid, convert each score into a T x 4 matrix of MIDI pitches or special tokens, build train/val/test splits, and save processed data. Include robust error handling and skip scores that cannot be parsed cleanly.
```

### Prompt 2: MIDI output

```text
Implement src/midi_io.py. Write a function satb_matrix_to_midi(sequence, output_path, grid=0.5, tempo=90) that saves a T x 4 SATB pitch matrix as a MIDI file with four separate tracks or instruments. Handle rests and held notes cleanly.
```

### Prompt 3: Task 1 model

```text
Implement a PyTorch SATBGRULanguageModel for unconditioned chorale generation. Input shape is [batch, time, 4]. Use separate embeddings per voice or one shared embedding, concatenate voice embeddings, pass through a GRU, and produce four output heads for soprano, alto, tenor, and bass. Also implement training, validation loss, checkpoint saving, and temperature sampling.
```

### Prompt 4: Task 2 model

```text
Implement a PyTorch SopranoConditionedHarmonizer. Input is a soprano token sequence [batch, time]. Use an embedding layer and bidirectional GRU, then output alto, tenor, and bass logits for every time step. Include training, validation loss, pitch accuracy per voice, checkpoint saving, and inference that writes symbolic_conditioned.mid.
```

### Prompt 5: Baselines

```text
Implement two baselines: a MarkovSATBBaseline for unconditioned generation over SATB time-step tuples, and a LookupHarmonizationBaseline that predicts alto, tenor, and bass from soprano pitch class and beat position with backoff. Include fit, predict/sample, and evaluation methods.
```

### Prompt 6: Evaluation metrics

```text
Implement music-theory-inspired metrics for SATB sequences: voice range violation rate, voice crossing count, consonance rate, repeated-note ratio, chord diversity, parallel fifth/octave count, pitch-class histogram, and histogram distance. Include docstrings and unit-test-like examples.
```

### Prompt 7: Notebook assembly

```text
Create a clean Jupyter notebook for the BachBot project. It should include project overview, dataset loading, EDA plots, preprocessing explanation, Task 1 modeling/evaluation, Task 2 modeling/evaluation, related work, limitations, and links to generated MIDI outputs. The notebook should be readable after export to HTML and should not require peer graders to run the code.
```

---

## 17. Example `configs/mvp.yaml`

```yaml
seed: 42

data:
  grid: 0.5
  max_seq_len: 256
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  max_chorales: null

vocab:
  min_pitch: 21
  max_pitch: 108
  special_tokens:
    - PAD
    - REST
    - HOLD
    - START
    - END

training:
  batch_size: 16
  epochs: 20
  learning_rate: 0.001
  weight_decay: 0.00001
  grad_clip: 1.0
  device: auto

unconditioned_model:
  embedding_dim: 64
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2
  temperature: 1.0
  sample_length: 128

conditioned_model:
  embedding_dim: 64
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2
  bidirectional: true

paths:
  processed_data: outputs/processed/bach_chorales.pkl
  figures_dir: outputs/figures
  tables_dir: outputs/tables
  midi_dir: outputs/midi
  checkpoints_dir: outputs/checkpoints
```

---

## 18. Submission Sanity Checks

Before submission, verify:

```bash
ls -lh submission/
```

Expected files:

```text
workbook.html
video_url.txt
symbolic_unconditioned.mid
symbolic_conditioned.mid
```

Check that the notebook export starts with a valid HTML doctype:

```bash
head -n 1 submission/workbook.html
```

Expected:

```text
<!DOCTYPE html>
```

Check that `video_url.txt` contains only one line:

```bash
cat submission/video_url.txt
```

Check MIDI files are not empty:

```bash
ls -lh submission/*.mid
```

---

## 19. Final Presentation Takeaways

The final presentation should make these points clear:

1. The project completed two required symbolic generation tasks.
2. Both tasks used the same Bach chorale dataset and SATB representation.
3. Task 1 generated full chorales from scratch.
4. Task 2 generated lower-voice harmonizations from a soprano melody.
5. The project compared baselines with neural models.
6. The evaluation included both ML metrics and musical metrics.
7. The generated MIDI files were successfully produced and can be listened to.
8. Limitations were analyzed honestly.

---

## 20. Definition of Done

The project is done when:

```text
1. Running the pipeline produces both final MIDI files.
2. The notebook contains all results needed for peer grading.
3. The notebook is exported as workbook.html.
4. The presentation video explains the code and results clearly.
5. The submission folder contains all required files with exact names.
6. Each group member can explain their part and answer basic questions.
```
