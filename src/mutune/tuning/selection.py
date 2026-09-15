"""Constraint-aware acquisition with a continuous feasibility probability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .history import EvaluationRecord, candidate_key
from .surrogate import CandidatePrediction

_SQRT_TWO = math.sqrt(2.0)
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / _SQRT_TWO))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / _SQRT_TWO_PI


@dataclass(frozen=True, slots=True)
class AcquisitionScore:
    prediction: CandidatePrediction
    probability_feasible: float
    expected_improvement: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "probability_feasible": self.probability_feasible,
            "expected_improvement": self.expected_improvement,
            "score": self.score,
        }


class ConstraintAwareAcquisition:
    """Feasibility-weighted expected improvement for a maximized objective."""

    def __init__(
        self,
        *,
        objective_metric: str,
        constraint_metric: str,
        constraint_threshold: float,
        exploration_weight: float = 0.2,
    ) -> None:
        if exploration_weight < 0:
            raise ValueError("exploration_weight cannot be negative")
        self.objective_metric = objective_metric
        self.constraint_metric = constraint_metric
        self.constraint_threshold = constraint_threshold
        self.exploration_weight = exploration_weight

    def best_feasible(self, records: Sequence[EvaluationRecord]) -> float | None:
        values = [
            record.metrics[self.objective_metric]
            for record in records
            if record.ok
            and self.objective_metric in record.metrics
            and self.constraint_metric in record.metrics
            and record.metrics[self.constraint_metric] >= self.constraint_threshold
        ]
        return max(values, default=None)

    def probability_feasible(self, prediction: CandidatePrediction) -> float:
        sigma = max(prediction.constraint.stddev, 1e-12)
        z_value = (prediction.constraint.mean - self.constraint_threshold) / sigma
        # Keep a strictly probabilistic value even in floating-point tails.  This
        # avoids the legacy one-prediction path degenerating to a 0/1 gate.
        return min(1.0 - 1e-12, max(1e-12, normal_cdf(z_value)))

    @staticmethod
    def expected_improvement(prediction: CandidatePrediction, incumbent: float) -> float:
        sigma = max(prediction.objective.stddev, 1e-12)
        delta = prediction.objective.mean - incumbent
        z_value = delta / sigma
        return max(0.0, delta * normal_cdf(z_value) + sigma * normal_pdf(z_value))

    def rank(
        self,
        predictions: Sequence[CandidatePrediction],
        records: Sequence[EvaluationRecord],
    ) -> list[AcquisitionScore]:
        incumbent = self.best_feasible(records)
        if not predictions:
            return []

        cold_utilities = [
            prediction.objective.mean + self.exploration_weight * prediction.objective.stddev
            for prediction in predictions
        ]
        low = min(cold_utilities)
        high = max(cold_utilities)

        ranked: list[AcquisitionScore] = []
        for prediction, cold_utility in zip(predictions, cold_utilities, strict=True):
            probability = self.probability_feasible(prediction)
            if incumbent is None:
                normalized_utility = (
                    1.0 if math.isclose(high, low) else (cold_utility - low) / (high - low)
                )
                expected_improvement = max(normalized_utility, 0.0)
                # Feasibility remains the dominant signal until a feasible point
                # exists, with objective quality breaking probability ties.
                score = probability * (1.0 + expected_improvement)
            else:
                expected_improvement = self.expected_improvement(prediction, incumbent)
                score = probability * (
                    expected_improvement + self.exploration_weight * prediction.objective.stddev
                )
            ranked.append(
                AcquisitionScore(
                    prediction=prediction,
                    probability_feasible=probability,
                    expected_improvement=expected_improvement,
                    score=score,
                )
            )

        return sorted(
            ranked,
            key=lambda item: (
                -item.score,
                -item.probability_feasible,
                candidate_key(item.prediction.candidate),
            ),
        )

    def select(
        self,
        predictions: Sequence[CandidatePrediction],
        records: Sequence[EvaluationRecord],
        count: int,
    ) -> list[AcquisitionScore]:
        if count < 0:
            raise ValueError("selection count cannot be negative")
        return self.rank(predictions, records)[:count]


__all__ = [
    "AcquisitionScore",
    "ConstraintAwareAcquisition",
    "normal_cdf",
    "normal_pdf",
]
