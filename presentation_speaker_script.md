# BachBot Presentation Speaker Script

Suggested total length: about 20 minutes, plus optional MIDI playback.

Use this as a spoken guide rather than a word-for-word requirement. The best delivery will sound natural, but these notes keep the presentation aligned with the slides, notebook, code, and metrics.

## Slide 1: BachBot

Time: about 1 minute  
Speaker: Member A

Hi everyone, our project is called BachBot. The goal is to build a complete symbolic music-generation pipeline around Bach-style four-part chorales.

We chose two related assignment tasks. The first is symbolic unconditioned generation: the model learns a distribution over complete soprano, alto, tenor, and bass sequences, then generates a new chorale from scratch. The second is symbolic conditioned generation: given a soprano melody, the model generates the alto, tenor, and bass parts.

The important theme is that both tasks share the same data representation and evaluation pipeline. We compare simple statistical baselines with lightweight neural models, and we evaluate the outputs using both machine-learning metrics and music-theory-inspired checks. At the end, we will point to the final generated MIDI files and, if time allows, play both outputs.

Transition: First, I will quickly walk through how the talk is organized.

## Slide 2: Talk Roadmap

Time: about 1 minute  
Speaker: Member A

This roadmap follows the assignment rubric. We start with the task setup and motivation, then we cover the shared data pipeline and exploratory analysis once, because both tasks use the same SATB representation.

After that, we split into the two modeling sections. Task 1 is the unconditioned generator, where we compare a first-order Markov baseline against an autoregressive GRU. Task 2 is the harmonization task, where we compare a lookup-table baseline against a bidirectional GRU conditioned on the full soprano melody.

Then we discuss evaluation. We do not want to rely only on listening, because music quality is subjective and likelihood does not capture everything. So we use loss, perplexity or accuracy, and several music-theory-inspired metrics. Finally, we cover related work, limitations, and the final submitted files.

Transition: The first technical piece is the data representation.

## Slide 3: Data Representation

Time: about 1.5 minutes  
Speaker: Member A

Every model in this project sees a chorale as a time-indexed SATB token grid. The four columns are soprano, alto, tenor, and bass, and each row is one fixed time step.

We quantize the music at an eighth-note grid, which is 0.5 quarter length in the config. That gives us enough rhythmic detail for a small project while keeping the sequences manageable. Each cell is encoded as either a MIDI pitch token or a special token such as PAD, REST, HOLD, START, or END.

The slide also notes an important reproducibility detail. The code prefers the `music21` Bach chorale corpus, but in this run `music21` was not installed, so the pipeline used the deterministic chorale-style fallback corpus. That fallback is useful because it keeps the entire project runnable in a clean local environment, but it is also a limitation because it is not the full real Bach corpus.

For this run, the dataset has 96 sequences split into 67 train, 14 validation, and 15 test sequences. The average sequence length is 64 time steps, and the vocabulary has 93 tokens: 88 MIDI pitches plus 5 special symbols.

Transition: Once the representation is fixed, we can inspect whether the data looks musically reasonable.

## Slide 4: EDA

Time: about 1.5 minutes  
Speaker: Member A

These plots are the exploratory analysis that supports our preprocessing choices and evaluation design.

The sequence-length plot shows that the dataset is compact and regular enough for fixed-length batching. That matters because our neural models train on batched token sequences, so we need a maximum sequence length that does not throw away too much musical material.

The voice-range plot checks whether the soprano, alto, tenor, and bass parts sit in plausible SATB ranges. This becomes one of our evaluation ideas later: a generated sample can have good token likelihood but still be musically awkward if the bass jumps too high or voices cross.

The pitch-class histogram gives a compact view of the tonal footprint of the data. We use this later to compare generated samples against the training distribution. If the generated pitch-class distribution is very far from the reference, that is a signal that the sample may not sound like the training style.

Transition: Now that the data is encoded, we can model the first task: generating all four voices from scratch.

## Slide 5: Task 1 Modeling

Time: about 2 minutes  
Speaker: Member A

For Task 1, the problem is autoregressive SATB generation. At each time step, the model predicts the next four-voice state from the previous context.

Our baseline is a first-order Markov chain over complete SATB tuples. The state at time t is the full soprano, alto, tenor, and bass tuple. During training, the baseline counts transitions from one tuple to the next. At sampling time, it chooses the next tuple from the learned transition distribution. This is transparent and easy to implement, but the weakness is that it only sees one previous state and the complete four-voice state space is sparse.

The neural model is a four-head GRU language model. Each voice token gets embedded, the four embeddings are concatenated, and a two-layer GRU processes the sequence. Then the model has four separate output heads, one each for soprano, alto, tenor, and bass. The loss is the sum of cross-entropy losses across those four voices.

The reason this model is still lightweight is intentional. It is small enough to train locally for the class project, but expressive enough to capture more sequence memory than the Markov baseline.

Transition: Next we compare what this model actually produced.

## Slide 6: Task 1 Results

Time: about 2 minutes  
Speaker: Member A

This slide shows the unconditioned training curve and the music-metric comparison.

The training curve decreases steadily over 8 epochs, which tells us the GRU is learning the token prediction objective. In the metrics table, the GRU has a per-voice perplexity of about 7.61. The Markov baseline has a much larger tuple perplexity, around 264, but we should be careful: the Markov baseline scores complete SATB tuples while the GRU uses four per-voice heads, so those losses are not perfectly apples-to-apples.

The more informative comparison is musical behavior. The Markov sample has a repeated-note ratio of about 97.6 percent and very low chord diversity, around 4.2 percent. That means it tends to get stuck or repeat local states. The GRU has much higher chord diversity and a much smaller pitch-class histogram distance: about 0.072 compared with 1.135 for the Markov baseline.

But the GRU is not automatically better on every musical rule. It produces 22 voice crossings and 23 parallel perfect fifths or octaves in the generated sample. So the takeaway is that the GRU is more varied and distributionally closer, but without explicit harmony constraints it can still make classical voice-leading mistakes.

Transition: The second task uses the same representation, but the goal changes from free generation to harmonization.

## Slide 7: Task 2 Modeling

Time: about 2 minutes  
Speaker: Member B

Task 2 is soprano-conditioned harmonization. Here the soprano melody is known in advance, and the model generates the alto, tenor, and bass parts for each time step.

The lookup baseline uses simple conditioning features: soprano pitch class and beat position. For each combination, it predicts the most common lower-voice tuple from the training set. If there is no exact match, it backs off to broader counts. This baseline is useful because it tests whether simple local statistics already explain a lot of the harmonization.

The neural model is a bidirectional GRU harmonizer. It embeds the soprano token sequence, runs a two-layer bidirectional GRU, and predicts three output heads: alto, tenor, and bass. Bidirectionality is appropriate here because the whole soprano melody is known before we harmonize it. Looking ahead can help the model make lower voices that fit the phrase rather than only the current note.

The output is still symbolic, and the model is still trained from our data rather than using pretrained weights.

Transition: Now we can compare lookup harmonization against the BiGRU.

## Slide 8: Task 2 Results

Time: about 2 minutes  
Speaker: Member B

For the conditioned task, the BiGRU training curve also decreases over the 8 training epochs, so the neural model is learning the lower-voice prediction task.

In the metric table, the lookup baseline has an average lower-voice accuracy of about 27.0 percent, while the BiGRU improves that to about 29.2 percent. The improvement is modest, but it is meaningful because harmonization has many valid answers. Exact ATB accuracy is especially harsh: the BiGRU exact match is lower than the lookup baseline, but exact matching all three lower voices at once does not fully represent musical quality.

The music metrics tell a more balanced story. The BiGRU reduces the parallel fifth/octave count from 42 down to 12, which is a strong improvement for a voice-leading heuristic. It also has no range violations or voice crossings in this comparison. However, the cadence heuristic drops from 1.0 for the lookup baseline to 0.4 for the BiGRU, meaning the neural harmonizer does not always land on as convincing a final sonority.

So the main takeaway is: the BiGRU improves some predictive and voice-leading signals, but cadence and global phrase structure remain weak points.

Transition: These results are why we evaluate with more than one metric family.

## Slide 9: Evaluation Design

Time: about 2 minutes  
Speaker: Member C

This slide summarizes the evaluation philosophy. We separate predictive fit from musical plausibility because they answer different questions.

The ML-fit metrics tell us whether the model learned the token prediction task. For the unconditioned generator, that means loss and perplexity. For the harmonizer, that means per-voice accuracy and exact lower-voice accuracy.

The voice-leading metrics ask whether the generated SATB texture is singable and stylistically reasonable. We check range violations, voice crossings, repeated-note ratio, and parallel perfect fifths or octaves. These are not a complete music theory system, but they catch obvious failure modes.

The harmony metrics include strong-beat consonance, chord diversity, and a cadence heuristic. These help us distinguish between a sample that is stable but repetitive and a sample that is varied but harmonically messy.

Finally, the distribution metric compares pitch-class histograms. This gives us a compact way to ask whether the generated music still resembles the training data.

The key point is that no single metric decides whether the music is good. We need a combination of objective checks and listening.

Transition: That leads naturally into the listening demo.

## Slide 10: Listening Demo

Time: about 1.5 minutes, plus optional playback  
Speaker: Member C or all

For the demo, we suggest playing the two required MIDI outputs in this order. First, play `symbolic_unconditioned.mid`, which is the generated four-voice chorale from scratch. Then play `symbolic_conditioned.mid`, which is the harmonization of a soprano melody.

While listening, we want the audience to connect the audio back to the metrics. For the unconditioned output, listen for variety. Does it move beyond repeated states, or does it sound stuck? Also listen for voice-leading issues: do the voices feel like separate lines, or do parts cross awkwardly?

For the conditioned output, listen for how well the lower voices support the soprano. The harmonizer can be locally plausible, but the cadence may not always sound as final as a human-written Bach-style ending.

If there is extra time, the `outputs/midi` folder also contains baseline files and a reference test chorale. Those can be useful for comparison, but the two main submitted files are the symbolic unconditioned and symbolic conditioned MIDIs.

Transition: Before wrapping up, we want to situate this project against related work.

## Slide 11: Related Work And Limits

Time: about 2 minutes  
Speaker: Member B or C

The most directly relevant related work is DeepBach, because it focuses on Bach chorales and supports both generation and harmonization. Compared with DeepBach, our project is intentionally smaller. We use a simple grid representation, lightweight GRU models, and a class-scale training setup.

Music Transformer is another useful reference point. Transformers can model longer-range symbolic structure through self-attention, which could help with phrase-level planning and cadences. We chose GRUs because they are cheaper to train and easier to implement reliably for this assignment.

Bach chorales themselves are a standard benchmark because they are compact, symbolic, four-voice, harmonically regular, and interpretable with both ML metrics and music-theory heuristics.

The limitations are important. This run used a fallback corpus rather than the full `music21` Bach corpus. The grid is fixed at eighth-note resolution. Training is short, and the models do not enforce classical harmony constraints directly. Those limitations explain why the neural outputs can improve some metrics while still making musical mistakes.

Transition: The final slide summarizes what we achieved and what we would improve next.

## Slide 12: Wrap-Up

Time: about 1 minute  
Speaker: All or Member C

To wrap up, BachBot gives us a complete symbolic music-generation pipeline for two assignment tasks. The shared SATB token representation supports both unconditioned generation and soprano-conditioned harmonization.

The main result is not that the neural models are perfect. Instead, the result is more nuanced: the GRU and BiGRU improve some signals, like diversity, pitch-class similarity, average lower-voice accuracy, and parallel interval counts, but they still struggle with voice-leading and cadence quality.

For next steps, we would first run the same pipeline with the real `music21` Bach chorale corpus available. Then we would add stronger musical constraints, train longer, and potentially try a Transformer-style model for longer-range structure.

The final submission files are the exported workbook, the video URL placeholder, and the two generated MIDI files: `symbolic_unconditioned.mid` and `symbolic_conditioned.mid`.

Closing line: Thanks for watching. We will end by playing the generated examples.
