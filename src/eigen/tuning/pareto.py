"""Feasible non-dominated archives and normalized hypervolume utilities."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from eigen.config import MetricConstraint, ObjectiveSpec

from .history import EvaluationRecord


def feasible(metrics: Mapping[str, float], constraints: Sequence[MetricConstraint]) -> bool:
    return all(
        type(metrics.get(c.metric)) in (int, float)
        and math.isfinite(metrics[c.metric])
        and c.satisfied(dict(metrics))
        for c in constraints
    )


def utility_vector(
    metrics: Mapping[str, float], objectives: Sequence[ObjectiveSpec]
) -> tuple[float, ...]:
    return tuple(
        float(metrics[o.metric]) * (1 if o.direction == "maximize" else -1) for o in objectives
    )


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def archive(
    records: Sequence[EvaluationRecord],
    objectives: Sequence[ObjectiveSpec],
    constraints: Sequence[MetricConstraint],
) -> list[EvaluationRecord]:
    usable = [
        r
        for r in records
        if r.ok
        and feasible(r.metrics, constraints)
        and all(o.metric in r.metrics and math.isfinite(r.metrics[o.metric]) for o in objectives)
    ]
    vectors = [utility_vector(r.metrics, objectives) for r in usable]
    return [
        r
        for i, r in enumerate(usable)
        if not any(dominates(v, vectors[i]) for j, v in enumerate(vectors) if i != j)
    ]


def normalization_bounds(vectors: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    if not vectors:
        return ()
    # Put the reference point 5% below the observed range. Otherwise a frontier
    # of two trade-off endpoints has zero volume in both dimensions.
    return tuple(
        (min(column) - 0.05 * (max(column) - min(column)), max(column))
        for column in zip(*vectors, strict=True)
    )


def normalize_vector(
    vector: Sequence[float], bounds: Sequence[tuple[float, float]]
) -> tuple[float, ...]:
    # A constant objective provides no discrimination; all points receive 1.
    return tuple(
        1.0 if math.isclose(low, high) else max(0.0, min(1.0, (v - low) / (high - low)))
        for v, (low, high) in zip(vector, bounds, strict=True)
    )


def hypervolume(points: Sequence[Sequence[float]]) -> float:
    """Exact union of boxes from origin to nonnegative maximizing points.

    Recursive dimension sweep; intended for small archives (paper budgets),
    not for thousands of objectives. No external optimizer dependency.
    """
    if not points:
        return 0.0
    dimensions = len(points[0])
    if any(len(p) != dimensions or any(not math.isfinite(x) or x < 0 for x in p) for p in points):
        raise ValueError("hypervolume needs finite nonnegative vectors of equal dimension")
    if dimensions == 0:
        return 0.0
    if dimensions == 1:
        return max(p[0] for p in points)
    levels = sorted({0.0, *(p[-1] for p in points)})
    return sum(
        (upper - lower) * hypervolume([p[:-1] for p in points if p[-1] >= upper])
        for lower, upper in zip(levels, levels[1:], strict=False)
    )


def normalized_archive(
    records: Sequence[EvaluationRecord],
    objectives: Sequence[ObjectiveSpec],
    constraints: Sequence[MetricConstraint],
) -> tuple[list[EvaluationRecord], list[tuple[float, ...]]]:
    front = archive(records, objectives, constraints)
    vectors = [utility_vector(r.metrics, objectives) for r in front]
    bounds = normalization_bounds(vectors)
    return front, [normalize_vector(v, bounds) for v in vectors]
