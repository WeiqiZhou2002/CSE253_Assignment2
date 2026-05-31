# BachBot Model Upgrade Plan

**Project:** Bach-style symbolic music generation and melody-conditioned harmonization  
**Problem:** The current model can generate valid MIDI files, but some outputs sound ugly, unstable, or unmusical.  
**Goal:** Improve musical quality without rebuilding the whole project from scratch.

This file is designed for two uses:

1. **Codex implementation handoff:** concrete modules, functions, and pseudocode to generate code.
2. **Presentation planning:** a clear story for why the music improved and how to explain the upgrades.

---

## 1. Current project setup

The project has two symbolic generation tasks:

1. **Symbolic unconditioned generation**
   - Generate a complete 4-voice SATB chorale from scratch.
   - Output file: `symbolic_unconditioned.mid`

2. **Symbolic conditioned generation**
   - Given a soprano melody, generate alto, tenor, and bass.
   - Output file: `symbolic_conditioned.mid`

The current model is roughly:

- Unconditioned task: autoregressive GRU over SATB chord or voice tokens.
- Conditioned task: bidirectional GRU over soprano input, with output heads for lower voices.
- Evaluation: ML metrics plus musical metrics such as voice range violations, voice crossings, consonance, repeated notes, chord diversity, and cadence quality.

The main weakness is not only the neural architecture. The bigger issue is that raw likelihood-based generation does not directly optimize musical properties such as voice leading, harmonic stability, and cadences.

---

## 2. Main diagnosis: why the music sounds ugly

Likely causes:

1. **Weak or artificial training data**
   - If the project used a fallback corpus instead of real Bach chorales, the model may not have learned real harmonic patterns.

2. **Sampling is too random**
   - High temperature or unrestricted sampling often produces strange jumps and unstable chords.

3. **No musical constraints during decoding**
   - The model may output notes that are statistically plausible individually but bad together.

4. **Voice-leading errors**
   - Common problems:
     - alto above soprano
     - tenor above alto
     - bass too high
     - large jumps within one voice
     - parallel fifths or octaves

5. **Bad phrase ending**
   - Even a decent generated phrase can sound bad if it does not end on a stable cadence.

6. **MIDI writing may be too choppy**
   - If every grid step is written as a new note, repeated notes sound mechanical instead of sustained.

---

## 3. Upgrade priorities

Implement these in order. Do not jump immediately to a large Transformer unless the simpler upgrades are already working.

### Priority 1: use real Bach chorales

If `music21` is available, use its Bach chorale corpus instead of a deterministic fallback dataset.

```bash
pip install music21 pretty_midi
```

Recommended dataset pipeline:

1. Load Bach chorales from `music21.corpus`.
2. Extract soprano, alto, tenor, and bass parts.
3. Quantize to an eighth-note grid.
4. Convert every time step into SATB MIDI pitches.
5. Split into train, validation, and test sets.
6. Use only training data for augmentation.

Presentation story:

> Our first generated outputs were unstable partly because the fallback corpus was too small and artificial. We improved the training data by using real Bach chorales, which better represent SATB ranges, cadences, and voice-leading patterns.

---

### Priority 2: safer sampling

For unconditioned generation, use low-temperature top-k sampling.

Recommended settings:

```python
TEMPERATURE = 0.7
TOP_K = 5
```

Try this grid for experiments:

```python
temperatures = [0.6, 0.7, 0.8, 1.0]
top_ks = [3, 5, 8]
```

Avoid for final output:

```python
temperature > 1.2
```

High temperature can increase variety, but small music models often become chaotic.

#### Codex function request

Ask Codex to implement:

```python
def sample_top_k_temperature(logits, temperature=0.7, top_k=5):
    """
    Sample one token from logits using temperature scaling and top-k filtering.

    Args:
        logits: 1D torch.Tensor of unnormalized logits.
        temperature: float. Lower values make sampling more conservative.
        top_k: int. Only sample from the top k tokens.

    Returns:
        int token id.
    """
```

---

### Priority 3: rule-based SATB postprocessing

Add a postprocessing step after generation.

Target function:

```python
def postprocess_satb(sequence, key="C"):
    """
    Clean generated SATB output.

    Args:
        sequence: list of [soprano, alto, tenor, bass] MIDI pitches.
        key: tonal center used for final cadence rule.

    Returns:
        cleaned sequence in the same format.
    """
```

Use these approximate SATB ranges:

```python
VOICE_RANGES = {
    "soprano": (60, 81),  # C4 to A5
    "alto":    (55, 74),  # G3 to D5
    "tenor":   (48, 67),  # C3 to G4
    "bass":    (40, 60),  # E2 to C4
}
```

Postprocessing rules:

1. **Clamp pitches by octave**
   - If a pitch is too high, subtract 12 until it fits.
   - If a pitch is too low, add 12 until it fits.

2. **Fix voice crossing**
   - Enforce:

```text
soprano >= alto >= tenor >= bass
```

3. **Reduce huge melodic jumps**
   - If a voice jumps by more than 12 semitones, try shifting the current note by an octave.

4. **Force stable ending**
   - For C major demo outputs, a safe final chord is:

```python
final_chord = [64, 60, 55, 48]  # E4, C4, G3, C3
```

or:

```python
final_chord = [60, 55, 52, 48]  # C4, G3, E3, C3
```

5. **Merge repeated notes when writing MIDI**
   - Repeated grid pitches should become sustained notes, not separate re-attacked notes.

---

## 4. Candidate generation and reranking

This is the highest-impact upgrade after real data.

Instead of generating one output, generate many candidates, clean them, score them, and save the best one.

Target workflow:

```python
candidates = []

for seed in range(50):
    raw_seq = generate_unconditioned(model, seed=seed, temperature=0.7, top_k=5)
    clean_seq = postprocess_satb(raw_seq, key="C")
    metrics = evaluate_music_quality(clean_seq)
    score = musical_badness_score(metrics)
    candidates.append((score, clean_seq, metrics))

best_score, best_seq, best_metrics = min(candidates, key=lambda x: x[0])
write_satb_midi(best_seq, "symbolic_unconditioned.mid")
```

### Musical badness score

Lower is better.

```python
def musical_badness_score(metrics):
    return (
        4.0 * metrics["range_violation_rate"]
        + 5.0 * metrics["voice_crossing_rate"]
        + 3.0 * metrics["parallel_fifths_octaves_rate"]
        + 2.0 * metrics["strong_beat_dissonance_rate"]
        + 1.0 * metrics["large_leap_rate"]
        + 1.0 * metrics["repetition_penalty"]
        + 1.0 * metrics["pitch_histogram_distance"]
        - 0.5 * metrics["chord_diversity"]
        - 2.0 * metrics["cadence_score"]
    )
```

If some metrics are not implemented yet, start with this simpler score:

```python
def simple_badness_score(metrics):
    return (
        5.0 * metrics["voice_crossing_rate"]
        + 4.0 * metrics["range_violation_rate"]
        + 2.0 * metrics["strong_beat_dissonance_rate"]
        + 1.0 * metrics["large_leap_rate"]
        - 2.0 * metrics["cadence_score"]
    )
```

Presentation story:

> The neural model proposes multiple candidates. We then use a lightweight symbolic reranker to choose the candidate with the best voice-leading and harmonic heuristic score. This improves musical quality without pretending that likelihood alone captures all musical quality.

---

## 5. Upgrade for conditioned harmonization

The conditioned model should not simply choose the most likely alto, tenor, and bass independently at each time step.

Better approach:

```text
model probability + musical penalty = decoding objective
```

### Beam search with musical constraints

At each time step:

1. Use model logits to get top-k alto candidates.
2. Use model logits to get top-k tenor candidates.
3. Use model logits to get top-k bass candidates.
4. Try all candidate lower-voice combinations.
5. Add model negative log probability.
6. Add musical penalties.
7. Keep the best beam states.

Target function:

```python
def harmonize_with_beam_search(
    model,
    soprano_sequence,
    beam_size=8,
    top_k_per_voice=5,
    rule_weight=1.0,
):
    """
    Generate alto, tenor, and bass conditioned on soprano using beam search.

    Returns:
        list of [soprano, alto, tenor, bass] MIDI pitches.
    """
```

### Chord penalty function

```python
def chord_penalty(current_chord, previous_chord=None, beat_index=0):
    """
    Penalize bad SATB combinations.

    current_chord: [soprano, alto, tenor, bass]
    previous_chord: previous [soprano, alto, tenor, bass], or None
    beat_index: integer grid step
    """
```

Recommended penalties:

```text
+10 if voice crossing occurs
+5  if any voice is outside range
+3  if strong beat chord is highly dissonant
+2  if any voice leaps more than an octave
+2  if parallel fifth or octave occurs with previous chord
-2  if final chord is tonic-like
```

For the conditioned task, this is especially useful when using a familiar soprano melody such as an original C-major melody or the public-domain Ode to Joy melody.

---

## 6. Transposition augmentation

Bach chorale data is small. Use transposition augmentation only on the training split.

Suggested shifts:

```python
TRANSPOSE_SHIFTS = [-5, -3, -2, 0, 2, 3, 5]
```

Keep a transposed example only if all voices still fit SATB ranges.

Target function:

```python
def augment_by_transposition(sequences, shifts=(-5, -3, -2, 0, 2, 3, 5)):
    """
    Return augmented SATB sequences after transposition.
    Drop examples that violate SATB voice ranges.
    """
```

Presentation story:

> Because the Bach chorale corpus is relatively small, we used transposition augmentation to increase training examples while preserving musical structure. Validation and test sets were not augmented, so evaluation remains fair.

---

## 7. MIDI writing improvement

The generated MIDI should sound smooth.

Common bug:

```text
C4 C4 C4 C4
```

is written as four separate short notes.

Better:

```text
C4 held for four grid steps
```

Target function:

```python
def write_satb_midi(sequence, output_path, step_duration=0.5, tempo=90):
    """
    Write SATB MIDI and merge repeated notes into sustained durations.
    """
```

Algorithm:

1. For each voice separately:
   - scan through grid steps
   - detect runs of identical pitches
   - write one note per run
2. Skip rests if using a rest token.
3. Use different MIDI programs or channels if desired, but keep it simple.

This can make the generated output sound much less mechanical even if the pitch sequence is unchanged.

---

## 8. Training upgrades

These are useful but lower priority than decoding and postprocessing.

Recommended training improvements:

1. **Early stopping**
   - Save the model with the best validation loss.

2. **Gradient clipping**
   - Use `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.

3. **Learning-rate scheduling**
   - Reduce learning rate when validation loss plateaus.

4. **Dropout**
   - Add dropout between recurrent layers or before output heads.

5. **Separate pitch heads per voice**
   - If chord-token vocabulary is too sparse, predict each voice separately.

6. **Report validation curves**
   - Include loss/perplexity curves in the notebook and presentation.

Target training settings:

```python
BATCH_SIZE = 16
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
PATIENCE = 7
GRAD_CLIP = 1.0
```

---

## 9. Recommended code structure

Ask Codex to organize the project like this:

```text
project_root/
  workbook.ipynb
  workbook.html
  video_url.txt
  symbolic_unconditioned.mid
  symbolic_conditioned.mid

  src/
    data.py
    models.py
    train.py
    decode.py
    music_rules.py
    evaluate.py
    midi_io.py
    plots.py

  outputs/
    midi/
    figures/
    checkpoints/
    metrics/
```

### `src/data.py`

Responsibilities:

- load Bach chorales
- parse SATB voices
- quantize to eighth-note grid
- split train/validation/test
- transposition augmentation

Functions:

```python
load_bach_chorales()
chorale_to_satb_grid(score, step=0.5)
split_dataset(sequences, train=0.7, val=0.15, test=0.15, seed=0)
augment_by_transposition(sequences, shifts)
```

### `src/models.py`

Responsibilities:

- model definitions

Classes:

```python
class UnconditionedGRU(torch.nn.Module):
    pass

class ConditionalHarmonizationGRU(torch.nn.Module):
    pass
```

### `src/train.py`

Responsibilities:

- training loops
- validation loss
- checkpoint saving

Functions:

```python
train_unconditioned(...)
train_conditioned(...)
load_best_checkpoint(...)
```

### `src/decode.py`

Responsibilities:

- sampling
- top-k temperature decoding
- beam search
- candidate generation

Functions:

```python
sample_top_k_temperature(logits, temperature, top_k)
generate_unconditioned(model, ...)
harmonize_argmax(model, soprano_sequence)
harmonize_with_beam_search(model, soprano_sequence, ...)
generate_and_rerank(model, n_candidates=50)
```

### `src/music_rules.py`

Responsibilities:

- music theory heuristics
- SATB corrections
- penalty functions

Functions:

```python
is_in_voice_range(pitch, voice)
fix_pitch_range(pitch, voice)
fix_voice_crossing(chord)
limit_large_leap(prev_pitch, pitch, voice)
is_consonant_interval(interval)
count_parallel_fifths_octaves(sequence)
chord_penalty(current_chord, previous_chord=None, beat_index=0)
postprocess_satb(sequence, key="C")
```

### `src/evaluate.py`

Responsibilities:

- ML metrics
- musical metrics
- candidate scoring

Functions:

```python
evaluate_ml_metrics(model, dataset)
evaluate_music_quality(sequence)
musical_badness_score(metrics)
compare_baselines_and_models(results)
```

### `src/midi_io.py`

Responsibilities:

- MIDI export
- note merging

Functions:

```python
write_satb_midi(sequence, output_path, step_duration=0.5, tempo=90)
write_melody_midi(melody, output_path, step_duration=0.5, tempo=90)
```

### `src/plots.py`

Responsibilities:

- EDA and result figures

Functions:

```python
plot_pitch_ranges(...)
plot_length_distribution(...)
plot_training_curves(...)
plot_metric_comparison(...)
```

---

## 10. Codex task prompts

Use these as direct instructions for Codex.

### Prompt 1: safer sampling

```text
Implement sample_top_k_temperature(logits, temperature=0.7, top_k=5) in src/decode.py. It should accept a 1D torch tensor of logits, apply temperature scaling, keep only the top k logits, convert to probabilities with softmax, and return one sampled token id. Include input validation and make it deterministic when a torch random seed is set.
```

### Prompt 2: SATB postprocessing

```text
Implement postprocess_satb(sequence, key="C") in src/music_rules.py. The input is a list of [soprano, alto, tenor, bass] MIDI pitches. Use SATB ranges, fix out-of-range notes by octave shifting, fix voice crossing, reduce melodic jumps larger than an octave, and force a stable final C-major chord when key="C". Return the cleaned sequence without modifying the input in place.
```

### Prompt 3: musical metrics and badness score

```text
Implement evaluate_music_quality(sequence) and musical_badness_score(metrics) in src/evaluate.py. Metrics should include range_violation_rate, voice_crossing_rate, large_leap_rate, strong_beat_dissonance_rate, parallel_fifths_octaves_rate if feasible, chord_diversity, and cadence_score. The badness score should be lower for more musical sequences.
```

### Prompt 4: generate many candidates and rerank

```text
Implement generate_and_rerank(model, n_candidates=50, temperature=0.7, top_k=5). It should generate multiple unconditioned SATB candidates, postprocess each one, evaluate each one, compute musical_badness_score, and return the best sequence plus a table of candidate metrics.
```

### Prompt 5: conditioned beam search

```text
Implement harmonize_with_beam_search(model, soprano_sequence, beam_size=8, top_k_per_voice=5, rule_weight=1.0). At each time step, combine the top-k alto, tenor, and bass predictions, score them with model negative log probability plus rule_weight times chord_penalty, keep the best beam states, and return the best SATB sequence.
```

### Prompt 6: MIDI note merging

```text
Update write_satb_midi(sequence, output_path, step_duration=0.5, tempo=90) so consecutive identical pitches in the same voice are written as one longer sustained MIDI note instead of repeated short notes.
```

---

## 11. Experiment matrix

Use this small experiment matrix for the notebook and presentation.

| Experiment | Data | Decoding | Postprocess | Rerank | Expected result |
|---|---|---|---|---|---|
| A | fallback or old data | argmax/sample | no | no | weakest, likely ugly |
| B | Bach chorales | argmax/sample | no | no | better style, still errors |
| C | Bach chorales | temperature 0.7, top-k 5 | yes | no | fewer obvious errors |
| D | Bach chorales | temperature 0.7, top-k 5 | yes | 50 candidates | best final output |
| E | Bach chorales + augmentation | temperature 0.7, top-k 5 | yes | 50 candidates | possible final if time |

For conditioned harmonization:

| Experiment | Melody input | Decoder | Expected result |
|---|---|---|---|
| A | test-set Bach soprano | argmax lower voices | baseline neural output |
| B | test-set Bach soprano | beam search + rules | cleaner harmonization |
| C | custom C-major melody | beam search + rules | presentation demo |
| D | Ode to Joy melody | beam search + rules | recognizable demo |

---

## 12. Evaluation table to include in notebook

Create a table like this:

| Method | Test loss | Perplexity | Range violation | Voice crossing | Strong-beat consonance | Large leap rate | Cadence score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random baseline | - | - | high | high | low | high | low |
| Markov/common-chord baseline | ... | ... | ... | ... | ... | ... | ... |
| GRU raw sampling | ... | ... | ... | ... | ... | ... | ... |
| GRU + postprocess | ... | ... | ... | ... | ... | ... | ... |
| GRU + postprocess + rerank | ... | ... | ... | ... | ... | ... | ... |

Important presentation point:

> The reranked model may not have lower perplexity than the raw model, because reranking changes the generated output after model sampling. But it should improve musical metrics such as range, voice crossing, consonance, and cadence quality.

---

## 13. Three-person group division

Since this is a three-person group project, split the upgrade work like this.

### Member 1: data and model training

Responsibilities:

- Install and test `music21` Bach chorale loading.
- Replace fallback corpus with real Bach chorales.
- Implement train/validation/test split.
- Add transposition augmentation.
- Train updated unconditioned and conditioned GRU models.
- Save checkpoints and training curves.

Deliverables:

- Updated dataset statistics.
- Training loss plots.
- Best model checkpoints.
- Notebook section: exploratory analysis and model training.

### Member 2: decoding, postprocessing, and MIDI output

Responsibilities:

- Implement temperature and top-k sampling.
- Implement SATB postprocessing.
- Implement candidate reranking.
- Implement conditioned beam search.
- Improve MIDI writing by merging repeated notes.
- Generate final MIDI files.

Deliverables:

- `symbolic_unconditioned.mid`
- `symbolic_conditioned.mid`
- Candidate comparison table.
- Notebook section: decoding strategy and generated examples.

### Member 3: evaluation and presentation

Responsibilities:

- Implement or clean up musical metrics.
- Compare raw model vs upgraded model.
- Create evaluation tables and plots.
- Write related work and discussion sections.
- Prepare slides and video script.
- Verify final submission filenames.

Deliverables:

- Evaluation table.
- Before/after plots.
- Presentation slides/script.
- Final `workbook.html` and `video_url.txt` check.

---

## 14. Notebook update checklist

The final notebook should include these sections.

### Introduction

- State the two tasks clearly.
- Explain why Bach-style SATB is a good symbolic generation problem.
- Mention the upgrade motivation: raw model outputs were sometimes ugly.

### Dataset and preprocessing

- Dataset source.
- Number of chorales.
- Sequence length distribution.
- Pitch ranges by voice.
- Tokenization and quantization.
- Train/validation/test split.
- Data augmentation if used.

### Task 1: unconditioned generation

- Baseline model.
- GRU model.
- Sampling method.
- Postprocessing method.
- Candidate reranking.
- Generated MIDI example.
- Metrics table.

### Task 2: conditioned harmonization

- Input/output formulation.
- Baseline harmonizer.
- Conditional GRU model.
- Argmax vs beam search decoding.
- Custom melody or Ode to Joy input.
- Generated MIDI example.
- Metrics table.

### Evaluation and discussion

- Explain why perplexity is not enough.
- Discuss musical metrics.
- Compare before vs after upgrades.
- Discuss limitations honestly.

### Related work

Mention at least:

- DeepBach: Bach-style chorale generation and harmonization.
- Music Transformer: long-range structure with self-attention.
- Your project is simpler but focuses on a complete train-generate-evaluate pipeline.

---

## 15. Presentation structure

Target length: about 20 minutes.

| Time | Section | Speaker suggestion |
|---:|---|---|
| 1 min | Project overview and two tasks | Member 3 |
| 3 min | Dataset and preprocessing | Member 1 |
| 4 min | Task 1 model and raw generation issue | Member 1 |
| 4 min | Decoding, postprocessing, and reranking upgrade | Member 2 |
| 4 min | Task 2 conditioned harmonization and beam search | Member 2 |
| 3 min | Evaluation results and before/after comparison | Member 3 |
| 1 min | Related work and limitations | Member 3 |
| End | Play generated examples | All |

Suggested narrative:

> Our initial GRU models could generate valid MIDI, but some examples sounded unstable. We diagnosed that the model objective, cross-entropy loss, does not directly optimize musical quality. We improved the system by using real Bach chorales, safer top-k temperature sampling, SATB postprocessing, and candidate reranking with music-theory-inspired metrics. For conditioned harmonization, we added beam search with musical penalties so the model generates lower voices that better respect range, voice ordering, consonance, and cadence structure.

---

## 16. Final deliverables checklist

Before submission, the project folder should contain:

```text
workbook.html
video_url.txt
symbolic_unconditioned.mid
symbolic_conditioned.mid
```

Important:

- The notebook must be exported as real HTML and should start with `<!DOCTYPE html>`.
- The two MIDI files should be submitted at top level, not only inside `outputs/midi/`.
- `video_url.txt` should contain exactly one shareable Google Drive or YouTube link.
- The notebook should be readable without requiring graders to run code.

---

## 17. Minimal upgrade if time is short

If time is very limited, do only these four upgrades:

1. Use real Bach chorales through `music21`.
2. Use `temperature=0.7` and `top_k=5` for sampling.
3. Add SATB postprocessing for range, voice crossing, large jumps, and final cadence.
4. Generate 50 candidates and choose the best using musical metrics.

This is the best quality improvement per hour of work.

---

## 18. Expected final claim

Use this as the conclusion in the notebook or presentation:

> The upgraded system does not make the model larger; instead, it combines a learned neural music model with symbolic musical constraints. This hybrid approach reduced obvious SATB errors and produced more convincing generated chorales and harmonizations. The results also show why music generation should be evaluated with both predictive metrics and musical-quality metrics.
