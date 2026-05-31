# BachBot Presentation Speaker Script

Target length: about 18-20 minutes, plus optional MIDI playback.

This script is written to match `bachbot-symbolic-chorales-upgraded.pptx`. It is meant as a speaking guide, not something you must read word for word.

## Slide 1: BachBot

Time: about 1 minute  
Speaker: Member A

Hi everyone, our project is called BachBot.

The goal is to generate Bach-style four-part chorales using symbolic music representation. In this project, every piece is represented as four voices: soprano, alto, tenor, and bass.

We worked on two related tasks. The first task is unconditioned generation, where the model generates all four voices from scratch. The second task is conditioned generation, where we give the model a soprano melody and ask it to generate the alto, tenor, and bass parts.

The main idea of the project is not just to train a neural model and sample from it directly. We found that raw generation can sound unstable, so the final system combines learned models with music-theory-inspired decoding rules.

Transition: I will first show the overall structure of the talk.

## Slide 2: Roadmap

Time: about 1 minute  
Speaker: Member A

The presentation follows the full pipeline.

First, we explain the data representation: how Bach chorales become SATB token sequences. Then we show exploratory analysis, including sequence lengths, voice ranges, and pitch-class distributions.

After that, we cover the two modeling tasks. For Task 1, we compare a Markov baseline with a GRU model for unconditioned generation. For Task 2, we compare a lookup baseline with a bidirectional GRU harmonizer, and we also use a DeepBach-inspired Gibbs sampler for the final conditioned MIDI.

Then we discuss evaluation. Since music quality is subjective, we use both machine-learning metrics and music-theory metrics. Finally, we play or describe the generated MIDI files and discuss limitations.

Transition: The first important design choice is the data format.

## Slide 3: Data

Time: about 1.5 minutes  
Speaker: Member A

All models in this project use the same symbolic representation.

We load real Bach chorales from the `music21` corpus. In the final run, the dataset contains 371 usable chorales. Each chorale is quantized to an eighth-note grid, so each time step represents 0.5 quarter notes.

After quantization, every piece becomes a matrix with four columns: soprano, alto, tenor, and bass. Each cell is encoded as a MIDI pitch token, plus a few special tokens such as PAD, REST, HOLD, START, and END.

One important implementation detail is that we use absolute offsets when reading from `music21`. If we accidentally use measure-local offsets, most notes get painted onto the beginning of the matrix and the rest of the piece becomes REST. We fixed this by flattening each part before reading offsets.

The original split is 260 training chorales, 56 validation chorales, and 55 test chorales. To increase the training data, we apply transposition augmentation only to the training split. After augmentation and range filtering, the training set has 358 sequences. Validation and test are not augmented, so evaluation remains fair.

Transition: Before training models, we checked whether this representation looked musically reasonable.

## Slide 4: EDA

Time: about 1.5 minutes  
Speaker: Member A

This slide shows three exploratory plots.

The first plot shows sequence lengths. The main point is that the chorales are compact enough for fixed-length batching, which makes training simpler.

The second plot shows pitch ranges for each voice. This is important because SATB music has expected ranges: soprano should generally be highest, bass should generally be lowest, and the voices should not cross too often.

The third plot shows the pitch-class distribution. This gives us a compact picture of the tonal style of the training data. Later, we compare generated samples against this distribution. If a generated piece has a very different pitch-class histogram, it may not sound close to the Bach chorale data.

Transition: Now we move to Task 1, generating a complete chorale from scratch.

## Slide 5: Task 1

Time: about 2 minutes  
Speaker: Member A

Task 1 is unconditioned SATB generation. The model does not receive a melody. It has to generate soprano, alto, tenor, and bass together.

Our baseline is a first-order Markov model over complete SATB tuples. At each time step, the state is the full four-note chord. The baseline learns transition probabilities from one SATB tuple to the next. This is simple and interpretable, but it only remembers one previous step, and the space of possible SATB tuples is sparse.

The neural model is a four-head GRU language model. It embeds the four voice tokens, processes the sequence with a two-layer GRU, and predicts the next soprano, alto, tenor, and bass tokens with separate output heads.

The most important upgrade is the decoding step. Instead of generating one sample and accepting it, we generate 30 candidates using low-temperature top-k sampling. Then we apply SATB postprocessing and rerank the candidates with a music-quality badness score.

That score penalizes things like range violations, voice crossings, parallel perfect intervals, dissonance on strong beats, and very poor pitch distribution.

Transition: The next slide shows how this affected the output.

## Slide 6: Task 1 Results

Time: about 2 minutes  
Speaker: Member A

The left plot shows that the GRU training and validation loss decrease over 8 epochs. So the model is learning the token prediction task.

The right plot compares musical metrics for the unconditioned output. The reranked GRU sample has zero range violations and zero voice crossings. It also has slightly higher chord diversity than the Markov sample, about 0.73 compared with about 0.66.

The GRU output has per-voice perplexity around 4.35. The Markov baseline has tuple perplexity around 205, but the comparison is not perfectly apples-to-apples because the Markov model scores full SATB tuples while the GRU predicts voices separately.

The main takeaway is that after fixing the dataset, this task is much harder and more honest. Candidate generation plus symbolic reranking still helps avoid obvious SATB failures, but the generated music is not perfect.

Transition: Task 2 uses the same SATB representation, but the model is conditioned on a melody.

## Slide 7: Task 2

Time: about 2 minutes  
Speaker: Member B

Task 2 is soprano-conditioned harmonization.

Here, the soprano melody is fixed. The model's job is to generate the lower three voices: alto, tenor, and bass.

For the final conditioned decoder, we do not allow REST as an output for the lower voices. The reason is that this task is four-part harmonization, not voice-activity detection. After fixing the offset bug, real REST tokens are very rare in the Bach chorale grid, so allowing REST would mostly give the model an easy way to generate silence instead of harmony.

The baseline is a lookup model. It uses soprano pitch class and beat position, then predicts the most common lower-voice tuple from the training data. This baseline is useful because Bach chorales have many repeated local patterns, so a simple local method can already be strong.

The neural model is a bidirectional GRU. Since the full soprano melody is known in advance, bidirectionality makes sense: the model can use both past and future melody context before choosing lower voices.

For decoding, we use beam search. Instead of picking alto, tenor, and bass independently at each time step, beam search considers combinations of lower voices and scores them using model probability plus musical penalties.

We also include a DeepBach-inspired Gibbs sampler. It is not the original DeepBach model, but it follows the same idea of repeatedly resampling voices using local conditional distributions learned from Bach chorales.

Transition: Now we compare the conditioned results.

## Slide 8: Task 2 Results

Time: about 2 minutes  
Speaker: Member B

The conditioned GRU training curve also decreases, so the model is learning the lower-voice prediction task.

One interesting result is that the BiGRU now performs better than the lookup baseline on predictive accuracy. The lookup baseline has average voice accuracy around 0.23, while the BiGRU reaches about 0.39. Exact lower-voice accuracy is still low for both models, around 0.10 for lookup and 0.12 for BiGRU.

But exact match is not the whole story. For harmonization, there are many valid alto, tenor, and bass choices for the same soprano melody. A model can be musically acceptable even if it does not match the original Bach lower voices exactly.

The final conditioned MIDI uses the DeepBach-inspired Gibbs sampler. On the final conditioned output, the strong-beat consonance is about 99 percent, with zero range violations and zero voice crossings. It also avoids the long repeated-note problem we saw earlier.

The main lesson is that decoding strategy matters a lot. Cross-entropy training gives us probabilities, but beam search, Gibbs sampling, and symbolic penalties make the generated music more listenable.

Transition: Because of that, we evaluate the music with several metric families.

## Slide 9: Evaluation

Time: about 2 minutes  
Speaker: Member C

This slide summarizes our evaluation design.

We divide the metrics into four groups.

The first group is predictive fit. For the unconditioned task, this means loss and perplexity. For the conditioned task, this means per-voice accuracy and exact lower-voice accuracy.

The second group is voice leading. These metrics check whether the voices behave like singable SATB lines. We count range violations, voice crossings, repeated-note ratio, and parallel perfect fifths or octaves.

The third group is harmony. We check strong-beat consonance, chord diversity, and a cadence heuristic. These metrics help catch outputs that are either too unstable or too repetitive.

The fourth group is distribution matching. We compare the generated pitch-class histogram with the training data pitch-class histogram.

The key point is that no single metric is enough. Music generation needs both objective checks and listening.

Transition: That brings us to the listening demo.

## Slide 10: Listening Demo

Time: about 1.5 minutes, plus playback  
Speaker: Member C or all

For the demo, we play the two required MIDI files.

First, play `submission/symbolic_unconditioned.mid`. This is the piece generated from scratch. While listening, focus on whether the four voices move with some variety and whether the result avoids obvious voice crossing or range problems.

Second, play `submission/symbolic_conditioned.mid`. This is the harmonized version of the custom soprano melody. While listening, focus on whether the lower voices support the melody and whether they avoid staying on one note for too long.

The MIDI uses separate instruments to make the lines easier to hear: flute for soprano, oboe for alto, cello for tenor, and contrabass for bass.

If there is extra time, we can also compare against the baseline MIDI files or the reference test chorale in `outputs/midi`.

Transition: Before the conclusion, we briefly connect this project to related work.

## Slide 11: Related Work And Limits

Time: about 2 minutes  
Speaker: Member B or C

The most relevant related work is DeepBach. DeepBach also focuses on Bach chorales and supports both generation and harmonization. Our project is much smaller, but our Gibbs-style sampler is inspired by the same idea: repeatedly update voices using learned local musical context.

Another related model is Music Transformer. Transformers can model longer-range symbolic structure using self-attention. That could help with phrase-level planning and stronger cadences. We used GRUs because they are cheaper to train and easier to implement reliably for this class project.

Bach chorales are a useful benchmark because they are compact, symbolic, four-voice, and easy to evaluate with music-theory heuristics.

There are still limitations. The rhythm grid is fixed at eighth-note resolution. The GRU models are small. And our symbolic rules catch obvious errors, but they are not a complete theory of Bach harmony.

For future work, we would train longer, add stronger key-aware harmonic constraints, and try a Transformer-style model for longer-range structure.

Transition: The last slide summarizes the project.

## Slide 12: Wrap-Up

Time: about 1 minute  
Speaker: All or Member C

To wrap up, BachBot is a complete symbolic music-generation pipeline for two tasks: unconditioned SATB generation and soprano-conditioned harmonization.

The shared SATB token grid lets us use one data pipeline for both tasks. The main technical result is that learned models work better when combined with symbolic decoding constraints.

In Task 1, top-k sampling, postprocessing, and candidate reranking improve the final unconditioned sample. In Task 2, beam search and the DeepBach-inspired Gibbs sampler make the conditioned harmonization more stable and listenable.

The final submitted artifacts are the workbook and the two MIDI files: `symbolic_unconditioned.mid` and `symbolic_conditioned.mid`.

One final practical note: before submission, we still need to replace the placeholder in `video_url.txt` with the real public video link.

Closing line: Thank you. We will finish by playing the generated examples.
