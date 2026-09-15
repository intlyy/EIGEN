"""A lightweight mixed-space k-nearest-neighbour surrogate."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .history import EvaluationRecord


def _parameter_map(search_space: Any) -> Mapping[str, Any]:
    for owner in (
        getattr(search_space, "spec", None),
        getattr(search_space, "search_space", None),
        getattr(getattr(search_space, "profile", None), "search_space", None),
        search_space,
    ):
        parameters = getattr(owner, "parameters", None)
        if isinstance(parameters, Mapping):
            return parameters
    raise TypeError("search_space does not expose parameter specifications")


def _kind(spec: Any) -> str:
    value = getattr(spec, "kind", "categorical")
    return str(getattr(value, "value", value)).lower()


def _bounds(spec: Any) -> tuple[float, float] | None:
    value = getattr(spec, "bounds", None)
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        return float(value[0]), float(value[1])
    return None


@dataclass(frozen=True, slots=True)
class MetricPrediction:
    mean: float
    stddev: float


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    candidate: dict[str, Any]
    objective: MetricPrediction
    constraint: MetricPrediction
    nearest_distance: float
    neighbors: int

    def to_dict(self) -> dict[str, float]:
        return {
            "objective_mean": self.objective.mean,
            "objective_stddev": self.objective.stddev,
            "constraint_mean": self.constraint.mean,
            "constraint_stddev": self.constraint.stddev,
            "nearest_distance": self.nearest_distance,
            "neighbors": float(self.neighbors),
        }


class MixedSpaceKnnSurrogate:
    """Gower-style distance plus local weighted mean and uncertainty.

    Numeric dimensions are normalized by profile bounds; categorical and
    boolean dimensions use mismatch distance.  An inactive/active mismatch has
    maximal distance while two inactive values contribute nothing.
    """

    def __init__(
        self,
        search_space: Any,
        *,
        objective_metric: str,
        constraint_metric: str,
        k: int = 5,
    ) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        self.search_space = search_space
        self.objective_metric = objective_metric
        self.constraint_metric = constraint_metric
        self.k = k
        self.parameters = _parameter_map(search_space)
        self._records: list[EvaluationRecord] = []

    def fit(self, records: Sequence[EvaluationRecord]) -> "MixedSpaceKnnSurrogate":
        self._records = [
            record
            for record in records
            if record.ok
            and self.objective_metric in record.metrics
            and self.constraint_metric in record.metrics
            and math.isfinite(record.metrics[self.objective_metric])
            and math.isfinite(record.metrics[self.constraint_metric])
        ]
        return self

    def _active(self, candidate: Mapping[str, Any]) -> set[str]:
        try:
            return {str(name) for name in self.search_space.active_parameters(candidate)}
        except Exception:
            return {name for name in self.parameters if name in candidate}

    def distance(self, left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        left_active = self._active(left)
        right_active = self._active(right)
        components: list[float] = []
        for name, spec in self.parameters.items():
            in_left = name in left_active and name in left
            in_right = name in right_active and name in right
            if not in_left and not in_right:
                continue
            if in_left != in_right:
                components.append(1.0)
                continue

            left_value = left[name]
            right_value = right[name]
            if _kind(spec) in {"integer", "float"}:
                bounds = _bounds(spec)
                try:
                    difference = abs(float(left_value) - float(right_value))
                except (TypeError, ValueError):
                    components.append(1.0)
                    continue
                if bounds is None or math.isclose(bounds[0], bounds[1]):
                    components.append(0.0 if math.isclose(difference, 0.0) else 1.0)
                else:
                    components.append(min(1.0, difference / (bounds[1] - bounds[0])))
            else:
                components.append(0.0 if left_value == right_value else 1.0)
        return sum(components) / len(components) if components else 0.0

    @staticmethod
    def _global_scale(values: Sequence[float]) -> float:
        if not values:
            return 1.0
        spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        value_range = max(values) - min(values)
        magnitude = max(abs(statistics.fmean(values)), 1e-6)
        return max(spread, value_range / 2.0, magnitude * 0.10, 1e-6)

    @staticmethod
    def _estimate(
        neighbors: Sequence[tuple[float, EvaluationRecord]],
        metric: str,
        global_values: Sequence[float],
    ) -> MetricPrediction:
        if not neighbors:
            return MetricPrediction(
                mean=0.0,
                stddev=MixedSpaceKnnSurrogate._global_scale(global_values),
            )
        weights = [1.0 / max(distance, 1e-6) for distance, _ in neighbors]
        values = [record.metrics[metric] for _, record in neighbors]
        total_weight = sum(weights)
        mean = (
            sum(weight * value for weight, value in zip(weights, values, strict=True))
            / total_weight
        )
        local_variance = (
            sum(weight * (value - mean) ** 2 for weight, value in zip(weights, values, strict=True))
            / total_weight
        )
        scale = MixedSpaceKnnSurrogate._global_scale(global_values)
        nearest = neighbors[0][0]
        epistemic = nearest * scale
        floor = max(scale * 0.02, abs(mean) * 1e-4, 1e-9)
        return MetricPrediction(
            mean=mean,
            stddev=math.sqrt(max(local_variance, 0.0) + epistemic**2 + floor**2),
        )

    def predict(self, candidate: Mapping[str, Any]) -> CandidatePrediction:
        normalized = dict(candidate)
        distances = sorted(
            ((self.distance(normalized, record.candidate), record) for record in self._records),
            key=lambda pair: (pair[0], pair[1].sequence),
        )
        neighbor_count = min(self.k, len(distances))
        neighbors = distances[:neighbor_count]
        objective_values = [record.metrics[self.objective_metric] for record in self._records]
        constraint_values = [record.metrics[self.constraint_metric] for record in self._records]
        return CandidatePrediction(
            candidate=normalized,
            objective=self._estimate(neighbors, self.objective_metric, objective_values),
            constraint=self._estimate(neighbors, self.constraint_metric, constraint_values),
            nearest_distance=neighbors[0][0] if neighbors else 1.0,
            neighbors=neighbor_count,
        )

    def predict_many(self, candidates: Sequence[Mapping[str, Any]]) -> list[CandidatePrediction]:
        return [self.predict(candidate) for candidate in candidates]


__all__ = ["CandidatePrediction", "MetricPrediction", "MixedSpaceKnnSurrogate"]
