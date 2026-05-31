from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .metrics_music import CONSONANT_INTERVALS, VOICE_RANGES


VOICE_NAMES = ["soprano", "alto", "tenor", "bass"]


def _softmax_sample(scores: list[float], temperature: float, rng: np.random.Generator) -> int:
    """Sample an index from unnormalized log scores."""
    temp = max(float(temperature), 1e-6)
    arr = np.asarray(scores, dtype=np.float64) / temp
    arr -= np.max(arr)
    probs = np.exp(arr)
    probs = probs / probs.sum()
    return int(rng.choice(len(scores), p=probs))


class DeepBachGibbsSampler:
    """A small DeepBach-inspired Gibbs sampler trained from local chorales.

    DeepBach samples notes from conditional distributions given the rest of a
    chorale. This class keeps that spirit while staying lightweight: it learns
    empirical conditionals over soprano, beat position, vertical sonorities,
    and left/right SATB context, then repeatedly resamples lower voices.
    """

    def __init__(
        self,
        token_to_id: dict[Any, int],
        id_to_token: dict[int, Any],
        pad_id: int,
        steps_per_measure: int = 8,
        seed: int = 42,
        smoothing: float = 0.1,
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.pad_id = pad_id
        self.steps_per_measure = steps_per_measure
        self.seed = seed
        self.smoothing = smoothing
        self.state_counts: Counter[tuple[int, int, int, int]] = Counter()
        self.transition_counts: dict[tuple[int, int, int, int], Counter] = defaultdict(Counter)
        self.lower_by_soprano_beat: dict[tuple[int, int], Counter] = defaultdict(Counter)
        self.lower_by_soprano: dict[int, Counter] = defaultdict(Counter)
        self.lower_by_pc_beat: dict[tuple[int, int], Counter] = defaultdict(Counter)
        self.lower_by_pc: dict[int, Counter] = defaultdict(Counter)
        self.global_lower: Counter[tuple[int, int, int]] = Counter()

    def _pitch(self, token_id: int) -> int | None:
        token = self.id_to_token[int(token_id)]
        return int(token) if isinstance(token, int) else None

    def _pc(self, token_id: int) -> int:
        pitch = self._pitch(token_id)
        return 0 if pitch is None else pitch % 12

    def _nearest_token_for_pc(self, pc: int, target: int, low: int, high: int, ceiling: int | None = None) -> int | None:
        if ceiling is not None:
            high = min(high, ceiling)
        candidates = [
            pitch
            for pitch in range(low, high + 1)
            if pitch % 12 == pc and pitch in self.token_to_id
        ]
        if not candidates:
            return None
        pitch = min(candidates, key=lambda p: (abs(p - target), abs(p - (low + high) / 2)))
        return int(self.token_to_id[pitch])

    def fit(self, sequences: list[np.ndarray]) -> "DeepBachGibbsSampler":
        """Learn empirical conditionals from tokenized SATB sequences."""
        for seq in sequences:
            arr = np.asarray(seq, dtype=np.int64)
            rows = [tuple(map(int, row)) for row in arr if not np.any(row == self.pad_id)]
            self.state_counts.update(rows)
            for prev, curr in zip(rows[:-1], rows[1:]):
                self.transition_counts[prev][curr] += 1
            for t, row in enumerate(rows):
                soprano = row[0]
                lower = row[1:4]
                beat = t % self.steps_per_measure
                pc = self._pc(soprano)
                self.lower_by_soprano_beat[(soprano, beat)][lower] += 1
                self.lower_by_soprano[soprano][lower] += 1
                self.lower_by_pc_beat[(pc, beat)][lower] += 1
                self.lower_by_pc[pc][lower] += 1
                self.global_lower[lower] += 1
        return self

    def _log_counter_prob(self, counter: Counter, item: tuple[int, ...], vocab_size: int) -> float:
        total = sum(counter.values())
        return math.log((counter[item] + self.smoothing) / (total + self.smoothing * max(1, vocab_size)))

    def _reasonable_lower(self, soprano: int, lower: tuple[int, int, int], beat: int) -> bool:
        pitches = [self._pitch(token_id) for token_id in (soprano, *lower)]
        if any(p is None for p in pitches):
            return False
        soprano_pitch, alto, tenor, bass = [int(p) for p in pitches if p is not None]
        if not (soprano_pitch >= alto >= tenor >= bass):
            return False
        for voice, pitch in zip(VOICE_NAMES, [soprano_pitch, alto, tenor, bass]):
            low, high = VOICE_RANGES[voice]
            if pitch < low or pitch > high:
                return False
        strong_beat = beat in {0, self.steps_per_measure // 2}
        if strong_beat:
            pitches = [soprano_pitch, alto, tenor, bass]
            return all(
                abs(pitches[i] - pitches[j]) % 12 in CONSONANT_INTERVALS
                for i in range(4)
                for j in range(i + 1, 4)
            )
        return abs(soprano_pitch - bass) % 12 in CONSONANT_INTERVALS

    def _rule_based_lowers(self, soprano: int) -> list[tuple[int, int, int]]:
        soprano_pitch = self._pitch(soprano)
        if soprano_pitch is None:
            return []
        candidates: list[tuple[int, int, int]] = []
        soprano_pc = soprano_pitch % 12
        for bass_pitch in range(VOICE_RANGES["bass"][0], VOICE_RANGES["bass"][1] + 1):
            if abs(soprano_pitch - bass_pitch) % 12 not in CONSONANT_INTERVALS:
                continue
            for third in [4, 3]:
                chord_pcs = {bass_pitch % 12, (bass_pitch + third) % 12, (bass_pitch + 7) % 12}
                if soprano_pc not in chord_pcs:
                    continue
                alto_target = min(soprano_pitch - 3, 65)
                tenor_target = min(alto_target - 5, 57)
                alto_options = []
                tenor_options = []
                for pc in chord_pcs:
                    alto = self._nearest_token_for_pc(
                        pc,
                        alto_target,
                        VOICE_RANGES["alto"][0],
                        VOICE_RANGES["alto"][1],
                        ceiling=soprano_pitch - 1,
                    )
                    if alto is not None:
                        alto_options.append(alto)
                for alto in alto_options:
                    alto_pitch = self._pitch(alto)
                    if alto_pitch is None:
                        continue
                    for pc in chord_pcs:
                        tenor = self._nearest_token_for_pc(
                            pc,
                            tenor_target,
                            VOICE_RANGES["tenor"][0],
                            VOICE_RANGES["tenor"][1],
                            ceiling=alto_pitch - 1,
                        )
                        if tenor is None:
                            continue
                        tenor_pitch = self._pitch(tenor)
                        bass = int(self.token_to_id[bass_pitch])
                        if tenor_pitch is not None and tenor_pitch >= bass_pitch:
                            candidates.append((alto, tenor, bass))
        unique = list(dict.fromkeys(candidates))
        return unique[:24]

    def _candidate_lowers(self, soprano: int, beat: int, limit: int = 80) -> list[tuple[int, int, int]]:
        pc = self._pc(soprano)
        merged: Counter[tuple[int, int, int]] = Counter()
        for weight, counter in [
            (5, self.lower_by_soprano_beat.get((soprano, beat), Counter())),
            (3, self.lower_by_soprano.get(soprano, Counter())),
            (2, self.lower_by_pc_beat.get((pc, beat), Counter())),
            (1, self.lower_by_pc.get(pc, Counter())),
            (1, self.global_lower),
        ]:
            for item, count in counter.most_common(limit):
                merged[item] += weight * count
        candidates = [item for item, _ in merged.most_common(limit * 3)]
        rule_candidates = self._rule_based_lowers(soprano)
        filtered = [item for item in candidates if self._reasonable_lower(soprano, item, beat)]
        filtered.extend([item for item in rule_candidates if item not in filtered])
        return (filtered or rule_candidates or candidates)[:limit]

    def _transition_score(
        self,
        prev_state: tuple[int, int, int, int] | None,
        state: tuple[int, int, int, int],
        next_state: tuple[int, int, int, int] | None,
    ) -> float:
        score = 0.0
        vocab = max(1, len(self.state_counts))
        if prev_state is not None:
            score += 1.8 * self._log_counter_prob(self.transition_counts.get(prev_state, Counter()), state, vocab)
        if next_state is not None:
            score += 1.3 * self._log_counter_prob(self.transition_counts.get(state, Counter()), next_state, vocab)
        score += 0.3 * self._log_counter_prob(self.state_counts, state, vocab)
        return score

    def _vertical_score(self, state: tuple[int, int, int, int], beat: int) -> float:
        soprano = state[0]
        lower = state[1:4]
        pc = self._pc(soprano)
        candidate_count = max(1, len(self.global_lower))
        score = 0.0
        score += 2.0 * self._log_counter_prob(self.lower_by_soprano_beat.get((soprano, beat), Counter()), lower, candidate_count)
        score += 1.0 * self._log_counter_prob(self.lower_by_soprano.get(soprano, Counter()), lower, candidate_count)
        score += 0.8 * self._log_counter_prob(self.lower_by_pc_beat.get((pc, beat), Counter()), lower, candidate_count)
        return score

    def _parallel_penalty(
        self,
        prev_state: tuple[int, int, int, int] | None,
        state: tuple[int, int, int, int],
    ) -> float:
        if prev_state is None:
            return 0.0
        prev = [self._pitch(token_id) for token_id in prev_state]
        curr = [self._pitch(token_id) for token_id in state]
        if any(p is None for p in prev + curr):
            return 0.0
        penalty = 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                prev_interval = abs(int(prev[i]) - int(prev[j])) % 12
                curr_interval = abs(int(curr[i]) - int(curr[j])) % 12
                if prev_interval not in {0, 7} or curr_interval not in {0, 7}:
                    continue
                motion_i = int(curr[i]) - int(prev[i])
                motion_j = int(curr[j]) - int(prev[j])
                if motion_i == 0 or motion_j == 0:
                    continue
                if (motion_i > 0 and motion_j > 0) or (motion_i < 0 and motion_j < 0):
                    penalty -= 8.0
        return penalty

    def _music_score(
        self,
        prev_state: tuple[int, int, int, int] | None,
        state: tuple[int, int, int, int],
        next_state: tuple[int, int, int, int] | None,
        beat: int,
    ) -> float:
        pitches = [self._pitch(token_id) for token_id in state]
        if any(p is None for p in pitches):
            return -100.0
        soprano, alto, tenor, bass = [int(p) for p in pitches if p is not None]
        score = 0.0
        if not (soprano >= alto >= tenor >= bass):
            score -= 40.0
        for voice, pitch in zip(VOICE_NAMES, [soprano, alto, tenor, bass]):
            low, high = VOICE_RANGES[voice]
            if pitch < low or pitch > high:
                score -= 30.0

        strong_beat = beat in {0, self.steps_per_measure // 2}
        for i, upper in enumerate([soprano, alto, tenor]):
            consonant = abs(upper - bass) % 12 in CONSONANT_INTERVALS
            score += 2.2 if consonant else (-4.0 if strong_beat else -1.5)
            if i < 2:
                inner_consonant = abs([soprano, alto, tenor][i] - [alto, tenor, bass][i]) % 12 in CONSONANT_INTERVALS
                score += 0.7 if inner_consonant else -0.6

        for neighbor in [prev_state, next_state]:
            if neighbor is None:
                continue
            neighbor_pitches = [self._pitch(token_id) for token_id in neighbor]
            if any(p is None for p in neighbor_pitches):
                continue
            for voice_idx, (pitch, neighbor_pitch) in enumerate(zip(pitches, neighbor_pitches)):
                leap = abs(int(pitch) - int(neighbor_pitch))
                if leap <= 2:
                    score += 0.25
                elif leap <= 7:
                    score += 0.05
                elif voice_idx == 3 and leap <= 12:
                    score -= 0.15
                else:
                    score -= 0.8
                if leap == 0:
                    score -= 0.05
        score += self._parallel_penalty(prev_state, state)
        return score

    def _score_candidate(
        self,
        sequence: np.ndarray,
        t: int,
        lower: tuple[int, int, int],
    ) -> float:
        state = (int(sequence[t, 0]), *map(int, lower))
        prev_state = tuple(map(int, sequence[t - 1])) if t > 0 else None
        next_state = tuple(map(int, sequence[t + 1])) if t + 1 < len(sequence) else None
        beat = t % self.steps_per_measure
        return (
            self._vertical_score(state, beat)
            + self._transition_score(prev_state, state, next_state)
            + 4.0 * self._music_score(prev_state, state, next_state, beat)
        )

    def _best_initial_lower(self, soprano: int, beat: int) -> tuple[int, int, int]:
        candidates = self._candidate_lowers(soprano, beat, limit=20)
        if not candidates:
            return self.global_lower.most_common(1)[0][0]
        return candidates[0]

    def harmonize(
        self,
        soprano_sequence: np.ndarray,
        iterations: int = 5000,
        temperature: float = 0.65,
        candidate_limit: int = 80,
    ) -> np.ndarray:
        """Sample lower voices for a fixed soprano token sequence."""
        soprano = np.asarray(soprano_sequence, dtype=np.int64)
        if not len(soprano):
            raise ValueError("soprano_sequence is empty")
        rng = np.random.default_rng(self.seed)
        sequence = np.full((len(soprano), 4), self.pad_id, dtype=np.int64)
        sequence[:, 0] = soprano
        for t, token_id in enumerate(soprano):
            sequence[t, 1:4] = self._best_initial_lower(int(token_id), t % self.steps_per_measure)

        for _ in range(iterations):
            t = int(rng.integers(0, len(sequence)))
            candidates = self._candidate_lowers(int(sequence[t, 0]), t % self.steps_per_measure, limit=candidate_limit)
            if not candidates:
                continue
            scores = [self._score_candidate(sequence, t, lower) for lower in candidates]
            chosen = candidates[_softmax_sample(scores, temperature, rng)]
            sequence[t, 1:4] = chosen
        return sequence


def select_melodic_soprano(sequences: list[np.ndarray], id_to_token: dict[int, Any], target_length: int = 96) -> np.ndarray:
    """Choose a held-out soprano line with real melodic motion."""

    def pitches_for(seq: np.ndarray) -> list[int]:
        pitches: list[int] = []
        for token_id in seq[:target_length, 0]:
            token = id_to_token[int(token_id)]
            if isinstance(token, int):
                pitches.append(int(token))
        return pitches

    def score(seq: np.ndarray) -> float:
        pitches = pitches_for(seq)
        if len(pitches) < 8:
            return -1.0
        note_changes = sum(a != b for a, b in zip(pitches[:-1], pitches[1:]))
        unique = len(set(pitches))
        long_runs = 0
        run = 1
        for a, b in zip(pitches[:-1], pitches[1:]):
            if a == b:
                run += 1
            else:
                long_runs += max(0, run - 6)
                run = 1
        long_runs += max(0, run - 6)
        return unique * 2.0 + note_changes * 0.5 - long_runs

    best = max(sequences, key=score)
    return np.asarray(best[:target_length, 0], dtype=np.int64)
