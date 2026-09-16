"""Feasibility-weighted multiobjective acquisition with diverse batch selection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Sequence

from mutune.config import MetricConstraint, ObjectiveSpec

from .history import EvaluationRecord, candidate_key
from .llm_surrogate import LLMPrediction
from .pareto import archive, hypervolume, normalization_bounds, normalize_vector, utility_vector
from .surrogate import MixedSpaceKnnSurrogate


@dataclass(frozen=True)
class CALMScore:
    prediction: LLMPrediction
    score: float
    hypervolume_improvement: float
    diversity: float
    random_exploration: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability_feasible": self.prediction.probability_feasible,
            "score": self.score,
            "hypervolume_improvement": self.hypervolume_improvement,
            "diversity": self.diversity,
            "random_exploration": self.random_exploration,
        }


class ParetoBatchSelector:
    """Select using guidance coordinates (QPS/recall by default), not final rank."""

    def __init__(
        self,
        search_space: Any,
        objectives: Sequence[ObjectiveSpec],
        constraints: Sequence[MetricConstraint],
        *,
        rng: random.Random,
        exploration_probability: float = 0.1,
        diversity_weight: float = 0.2,
        uncertainty_weight: float = 0.2,
    ) -> None:
        self.objectives, self.constraints, self.rng = list(objectives), list(constraints), rng
        self.exploration_probability = exploration_probability
        self.diversity_weight, self.uncertainty_weight = diversity_weight, uncertainty_weight
        # Reuse only the canonical mixed-space distance, not the KNN predictor.
        self.distance = MixedSpaceKnnSurrogate(
            search_space, objective_metric="qps", constraint_metric="recall"
        ).distance

    def select(
        self, predictions: Sequence[LLMPrediction], records: Sequence[EvaluationRecord], count: int
    ) -> list[CALMScore]:
        if count < 0:
            raise ValueError("count must be nonnegative")
        if not predictions or count == 0:
            return []
        front = archive(records, self.objectives, self.constraints)
        measured = [utility_vector(r.metrics, self.objectives) for r in front]
        optimistic = [
            tuple(
                p.metrics[o.metric].mean * (1 if o.direction == "maximize" else -1)
                + self.uncertainty_weight * p.metrics[o.metric].stddev
                for o in self.objectives
            )
            for p in predictions
        ]
        bounds = normalization_bounds(measured + optimistic)
        known = [normalize_vector(v, bounds) for v in measured]
        vectors = [normalize_vector(v, bounds) for v in optimistic]
        base = hypervolume(known)
        improvements = [max(0.0, hypervolume([*known, v]) - base) for v in vectors]
        raw = [
            p.probability_feasible * (improvement + (0.0 if front else sum(v) / len(v)))
            for p, improvement, v in zip(predictions, improvements, vectors, strict=True)
        ]
        low, high = min(raw), max(raw)
        quality = [1.0 if math.isclose(low, high) else (x - low) / (high - low) for x in raw]
        remaining, selected = list(range(len(predictions))), []
        selected_indices: list[int] = []
        while remaining and len(selected) < count:
            diversity = {
                i: min(
                    (
                        self.distance(predictions[i].candidate, predictions[j].candidate)
                        for j in selected_indices
                    ),
                    default=1.0,
                )
                for i in remaining
            }
            scores = {i: quality[i] + self.diversity_weight * diversity[i] for i in remaining}
            explore = self.rng.random() < self.exploration_probability
            index = (
                self.rng.choice(remaining)
                if explore
                else min(
                    remaining, key=lambda i: (-scores[i], candidate_key(predictions[i].candidate))
                )
            )
            selected.append(
                CALMScore(
                    predictions[index],
                    scores[index],
                    improvements[index],
                    diversity[index],
                    explore,
                )
            )
            selected_indices.append(index)
            remaining.remove(index)
        return selected
