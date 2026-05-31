# BachBot Presentation Outline

Target length: about 20 minutes, plus optional MIDI playback.

| Slide | Time | Speaker | Section | Main point |
|---:|---:|---|---|---|
| 1 | 1 min | Member A | BachBot | Two symbolic tasks: free SATB generation and soprano-conditioned harmonization |
| 2 | 1 min | Member A | Roadmap | Data, models, results, evaluation, demo, limits |
| 3 | 1.5 min | Member A | Data representation | Real `music21` Bach chorales become eighth-note SATB token grids |
| 4 | 1.5 min | Member A | EDA | Sequence length, voice ranges, and pitch-class plots justify preprocessing and metrics |
| 5 | 2 min | Member A | Task 1 model | Markov baseline versus GRU with top-k sampling, postprocessing, and reranking |
| 6 | 2 min | Member A | Task 1 results | Reranking improves pitch-distribution match while keeping range and crossing errors at zero |
| 7 | 2 min | Member B | Task 2 model | Lookup baseline versus BiGRU beam search, plus DeepBach-inspired Gibbs decoding |
| 8 | 2 min | Member B | Task 2 results | Exact-match accuracy is not enough; symbolic decoding improves listenability |
| 9 | 2 min | Member C | Evaluation design | Predictive fit, voice leading, harmony, and distribution checks answer different questions |
| 10 | 1.5 min | Member C/all | Listening demo | Play `symbolic_unconditioned.mid`, then `symbolic_conditioned.mid` |
| 11 | 2 min | Member B/C | Related work and limits | DeepBach and Music Transformer motivate next steps beyond a class-scale GRU pipeline |
| 12 | 1 min | All | Wrap-up | Hybrid neural-symbolic decoding is the main takeaway; final files are ready except video URL |

Suggested playback order:

1. `submission/symbolic_unconditioned.mid`
2. `submission/symbolic_conditioned.mid`

Optional comparison files in `outputs/midi`: baseline MIDIs, reference test chorale, and conditioned soprano melody.
