from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np


def _weighted_choice(counter: Counter, rng: np.random.Generator, temperature: float = 1.0) -> tuple[int, ...]:
    items = list(counter.keys())
    weights = np.array([counter[item] for item in items], dtype=np.float64)
    if temperature <= 0:
        temperature = 1.0
    weights = np.power(weights, 1.0 / temperature)
    probs = weights / weights.sum()
    idx = int(rng.choice(len(items), p=probs))
    return tuple(items[idx])


class MarkovSATBBaseline:
    """First-order Markov baseline over complete SATB time-step tuples."""

    def __init__(self, pad_id: int = 0, smoothing: float = 1e-3, seed: int = 42) -> None:
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.seed = seed
        self.initial_counts: Counter = Counter()
        self.state_counts: Counter = Counter()
        self.transitions: dict[tuple[int, ...], Counter] = defaultdict(Counter)

    def fit(self, sequences: list[np.ndarray]) -> "MarkovSATBBaseline":
        for seq in sequences:
            states = [tuple(map(int, row)) for row in np.asarray(seq) if not np.any(row == self.pad_id)]
            if not states:
                continue
            self.initial_counts[states[0]] += 1
            self.state_counts.update(states)
            for prev, curr in zip(states[:-1], states[1:]):
                self.transitions[prev][curr] += 1
        return self

    def sample(self, length: int, temperature: float = 1.0) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        if not self.initial_counts:
            raise RuntimeError("baseline must be fit before sampling")
        state = _weighted_choice(self.initial_counts, rng, temperature)
        result = [state]
        for _ in range(length - 1):
            counter = self.transitions.get(state)
            if not counter:
                counter = self.state_counts
            state = _weighted_choice(counter, rng, temperature)
            result.append(state)
        return np.asarray(result, dtype=np.int64)

    def score(self, sequences: list[np.ndarray]) -> dict[str, float]:
        vocab_size = max(1, len(self.state_counts))
        total_log_prob = 0.0
        total_steps = 0
        for seq in sequences:
            states = [tuple(map(int, row)) for row in np.asarray(seq) if not np.any(row == self.pad_id)]
            for prev, curr in zip(states[:-1], states[1:]):
                counter = self.transitions.get(prev)
                if counter:
                    numerator = counter[curr] + self.smoothing
                    denominator = sum(counter.values()) + self.smoothing * vocab_size
                else:
                    numerator = self.state_counts[curr] + self.smoothing
                    denominator = sum(self.state_counts.values()) + self.smoothing * vocab_size
                total_log_prob += math.log(numerator / denominator)
                total_steps += 1
        loss = -total_log_prob / max(1, total_steps)
        return {"nll": loss, "perplexity": float(math.exp(min(loss, 20.0)))}


class LookupHarmonizationBaseline:
    """Predict lower voices from soprano pitch class and beat position."""

    def __init__(self, id_to_token: dict[int, Any], pad_id: int = 0, steps_per_measure: int = 8) -> None:
        self.id_to_token = id_to_token
        self.pad_id = pad_id
        self.steps_per_measure = steps_per_measure
        self.by_pc_beat: dict[tuple[Any, int], Counter] = defaultdict(Counter)
        self.by_pc: dict[Any, Counter] = defaultdict(Counter)
        self.global_counts: Counter = Counter()

    def _pitch_class(self, token_id: int) -> Any:
        token = self.id_to_token[int(token_id)]
        return int(token) % 12 if isinstance(token, int) else token

    def fit(self, sequences: list[np.ndarray]) -> "LookupHarmonizationBaseline":
        for seq in sequences:
            arr = np.asarray(seq)
            for t, row in enumerate(arr):
                if np.any(row == self.pad_id):
                    continue
                pc = self._pitch_class(int(row[0]))
                lower = tuple(map(int, row[1:4]))
                beat = t % self.steps_per_measure
                self.by_pc_beat[(pc, beat)][lower] += 1
                self.by_pc[pc][lower] += 1
                self.global_counts[lower] += 1
        return self

    @staticmethod
    def _most_common(counter: Counter) -> tuple[int, int, int]:
        return tuple(counter.most_common(1)[0][0])

    def predict(self, soprano_sequence: np.ndarray) -> np.ndarray:
        soprano = np.asarray(soprano_sequence)
        if soprano.ndim == 2:
            soprano = soprano[:, 0]
        output = np.full((len(soprano), 4), self.pad_id, dtype=np.int64)
        output[:, 0] = soprano
        for t, token_id in enumerate(soprano):
            pc = self._pitch_class(int(token_id))
            beat = t % self.steps_per_measure
            if self.by_pc_beat.get((pc, beat)):
                lower = self._most_common(self.by_pc_beat[(pc, beat)])
            elif self.by_pc.get(pc):
                lower = self._most_common(self.by_pc[pc])
            else:
                lower = self._most_common(self.global_counts)
            output[t, 1:4] = lower
        return output

    def score(self, sequences: list[np.ndarray]) -> dict[str, float]:
        correct = np.zeros(3, dtype=np.float64)
        total = 0.0
        exact = 0.0
        for seq in sequences:
            arr = np.asarray(seq)
            pred = self.predict(arr[:, 0])
            mask = ~np.any(arr == self.pad_id, axis=1)
            for row_true, row_pred in zip(arr[mask], pred[mask]):
                matches = row_true[1:4] == row_pred[1:4]
                correct += matches.astype(np.float64)
                exact += float(np.all(matches))
                total += 1.0
        total = max(total, 1.0)
        return {
            "alto_accuracy": float(correct[0] / total),
            "tenor_accuracy": float(correct[1] / total),
            "bass_accuracy": float(correct[2] / total),
            "average_voice_accuracy": float(correct.mean() / total),
            "exact_atb_accuracy": float(exact / total),
        }

