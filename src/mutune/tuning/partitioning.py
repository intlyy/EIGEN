"""Profile-driven conditional search-space regions."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from mutune.config import MetricConstraint, ObjectiveSpec
from mutune.utils import canonical_json, fingerprint, safe_name

from .history import EvaluationRecord
from .pareto import feasible, hypervolume, normalized_archive


def _search_spec(search_space: Any) -> Any:
    for attribute in ("spec", "search_space"):
        value = getattr(search_space, attribute, None)
        if value is not None and hasattr(value, "parameters"):
            return value
    profile = getattr(search_space, "profile", None)
    if profile is not None and hasattr(profile, "search_space"):
        return profile.search_space
    if hasattr(search_space, "parameters"):
        return search_space
    raise TypeError("search_space does not expose a profile SearchSpaceSpec")


def _parameters(search_space: Any) -> Mapping[str, Any]:
    parameters = getattr(_search_spec(search_space), "parameters", None)
    if not isinstance(parameters, Mapping):
        raise TypeError("search-space parameters must be a mapping")
    return parameters


def _explicit_partition_keys(spec: Any) -> list[str]:
    for attribute in ("partition_by", "region_parameters", "partition_keys"):
        value = getattr(spec, attribute, None)
        if value:
            return [str(item) for item in value]
    return []


def _predicate_parameter(predicate: Any) -> str | None:
    if isinstance(predicate, Mapping):
        value = predicate.get("parameter")
    else:
        value = getattr(predicate, "parameter", None)
    return str(value) if value else None


def _choice_values(parameter: Any, predicates: Iterable[Any]) -> tuple[Any, ...]:
    choices = getattr(parameter, "choices", None)
    if choices:
        return tuple(choices)
    kind = str(getattr(parameter, "kind", ""))
    if kind == "boolean":
        return (False, True)
    values: list[Any] = []
    for predicate in predicates:
        op = (
            predicate.get("op")
            if isinstance(predicate, Mapping)
            else getattr(predicate, "op", None)
        )
        value = (
            predicate.get("value")
            if isinstance(predicate, Mapping)
            else getattr(predicate, "value", None)
        )
        candidates = value if op in {"in", "not_in"} and isinstance(value, Sequence) else [value]
        for candidate in candidates:
            if candidate not in values:
                values.append(candidate)
    return tuple(values)


def _same_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _predicate_matches(predicate: Any, value: Any) -> bool:
    op = predicate.get("op") if isinstance(predicate, Mapping) else predicate.op
    expected = predicate.get("value") if isinstance(predicate, Mapping) else predicate.value
    if op == "eq":
        return _same_value(value, expected)
    if op == "ne":
        return not _same_value(value, expected)
    choices = expected if isinstance(expected, list) else []
    contained = any(_same_value(value, item) for item in choices)
    return contained if op == "in" else not contained


def _active_names(parameters: Mapping[str, Any], values: Mapping[str, Any]) -> set[str]:
    """Evaluate activation only, without coupling region discovery to constraints."""

    active_cache: dict[str, bool] = {}

    def is_active(name: str) -> bool:
        if name in active_cache:
            return active_cache[name]
        predicates = list(getattr(parameters[name], "active_if", ()) or ())
        active = all(
            is_active(str(_predicate_parameter(predicate)))
            and _predicate_matches(predicate, values[_predicate_parameter(predicate)])
            for predicate in predicates
        )
        active_cache[name] = active
        return active

    return {name for name in parameters if is_active(name)}


@dataclass(frozen=True, slots=True)
class Region:
    """A conditional slice defined entirely by profile parameter predicates."""

    id: str
    fixed: dict[str, Any] = field(default_factory=dict)
    active_parameters: tuple[str, ...] = ()

    def matches(self, candidate: Mapping[str, Any]) -> bool:
        return all(candidate.get(name) == value for name, value in self.fixed.items())


@dataclass(frozen=True, slots=True)
class RegionScore:
    region: Region
    score: float
    exploitation: float
    exploration: float
    probability_feasible: float
    observations: int


class ProfilePartitioner:
    """Derive regions from profile-declared conditional selector parameters."""

    def __init__(
        self,
        search_space: Any,
        *,
        runtime: Mapping[str, Any] | None = None,
        max_regions: int = 128,
    ) -> None:
        if max_regions < 1:
            raise ValueError("max_regions must be positive")
        self.search_space = search_space
        self.runtime = dict(runtime or {})
        self.max_regions = max_regions
        self._regions = self._build_regions()

    @property
    def regions(self) -> tuple[Region, ...]:
        return self._regions

    def _build_regions(self) -> tuple[Region, ...]:
        spec = _search_spec(self.search_space)
        parameters = _parameters(self.search_space)
        all_predicates: list[Any] = []
        predicates_by_selector: dict[str, list[Any]] = {}
        for parameter in parameters.values():
            predicates = list(getattr(parameter, "active_if", ()) or ())
            all_predicates.extend(predicates)
            for predicate in predicates:
                selector = _predicate_parameter(predicate)
                if selector:
                    predicates_by_selector.setdefault(selector, []).append(predicate)

        selectors = _explicit_partition_keys(spec) or sorted(predicates_by_selector)
        if not selectors:
            return (Region(id="all", active_parameters=tuple(parameters)),)

        choices_per_selector: list[tuple[Any, ...]] = []
        valid_selectors: list[str] = []
        for selector in selectors:
            if selector not in parameters:
                raise ValueError(f"profile partition selector is unknown: {selector}")
            choices = _choice_values(
                parameters[selector], predicates_by_selector.get(selector, all_predicates)
            )
            if not choices:
                continue
            valid_selectors.append(selector)
            choices_per_selector.append(choices)

        if not valid_selectors:
            return (Region(id="all", active_parameters=tuple(parameters)),)
        region_count = math.prod(len(choices) for choices in choices_per_selector)
        if region_count > self.max_regions:
            raise ValueError(
                f"profile expands to {region_count} regions; limit is {self.max_regions}"
            )

        regions: list[Region] = []
        seen_fixed: set[str] = set()
        for values in itertools.product(*choices_per_selector):
            proposed_fixed = dict(zip(valid_selectors, values, strict=True))
            activation_values = {
                name: proposed_fixed.get(name, parameter.default)
                for name, parameter in parameters.items()
            }
            active = _active_names(parameters, activation_values)
            # Nested selectors can themselves be inactive.  Dropping them folds
            # impossible Cartesian combinations into the parent region instead
            # of creating regions no canonical candidate can ever match.
            fixed = {name: value for name, value in proposed_fixed.items() if name in active}
            fixed_key = canonical_json(fixed)
            if fixed_key in seen_fixed:
                continue
            seen_fixed.add(fixed_key)
            label = ",".join(f"{name}={value}" for name, value in fixed.items())
            active_parameters = self._active_parameters(fixed)
            regions.append(
                Region(
                    id=safe_name(
                        f"{label}-{fingerprint(fixed, length=8)}",
                        max_length=100,
                    ),
                    fixed=fixed,
                    active_parameters=active_parameters,
                )
            )
        return tuple(regions)

    def _active_parameters(self, fixed: Mapping[str, Any]) -> tuple[str, ...]:
        try:
            # Canonicalization fills defaults for parameters made active by the
            # fixed selector and prunes every inactive branch.  Starting from a
            # random candidate could leave parameters from another index branch
            # in ``seed`` and incorrectly fall back to the full global space.
            canonical = self.search_space.canonicalize(
                dict(fixed),
                runtime=self.runtime,
                reject_inactive=False,
            )
            active = self.search_space.active_parameters(
                canonical,
                runtime=self.runtime,
            )
            return tuple(sorted(str(name) for name in active))
        except Exception:
            # The fixed selector values still define a valid region contract; the
            # proposer/renderer will perform authoritative candidate validation.
            return tuple(_parameters(self.search_space))

    def region_for(self, candidate: Mapping[str, Any]) -> Region:
        return next(
            (region for region in self._regions if region.matches(candidate)),
            self._regions[0],
        )

    def rank_regions(
        self,
        records: Sequence[EvaluationRecord],
        *,
        objective_metric: str,
        constraint_metric: str,
        threshold: float,
        exploration_weight: float,
        objectives: Sequence[ObjectiveSpec] | None = None,
        constraints: Sequence[MetricConstraint] | None = None,
        probes: Mapping[str, Sequence[Any]] | None = None,
        min_observations: int = 3,
    ) -> list[RegionScore]:
        """Pareto contribution for mature regions; predicted potential for new ones."""

        objectives = list(objectives or [ObjectiveSpec(metric=objective_metric)])
        constraints = list(
            constraints or [MetricConstraint(metric=constraint_metric, threshold=threshold)]
        )
        front, front_vectors = normalized_archive(records, objectives, constraints)
        volume = hypervolume(front_vectors)
        incumbent = max(
            (
                r.metrics[objective_metric]
                for r in records
                if r.ok and objective_metric in r.metrics and feasible(r.metrics, constraints)
            ),
            default=0.0,
        )

        grouped: dict[str, list[EvaluationRecord]] = {region.id: [] for region in self._regions}
        for record in records:
            if not record.ok:
                continue
            region = (
                next((item for item in self._regions if item.id == record.region_id), None)
                if record.region_id
                else None
            ) or self.region_for(record.candidate)
            grouped[region.id].append(record)

        scores: list[RegionScore] = []
        for region in self._regions:
            usable = grouped[region.id]
            count = len(usable)
            probability = (sum(feasible(r.metrics, constraints) for r in usable) + 1) / (count + 2)
            if count >= min_observations:
                other = [
                    v
                    for r, v in zip(front, front_vectors, strict=True)
                    if not region.matches(r.candidate)
                ]
                exploitation = max(0.0, volume - hypervolume(other))
                weighted_quality = exploitation
            else:
                potential = []
                for prediction in (probes or {}).get(region.id, ()):
                    estimate = prediction.metrics[objective_metric]
                    improvement = max(
                        0.0, estimate.mean - incumbent + exploration_weight * estimate.stddev
                    )
                    potential.append(
                        (prediction.probability_feasible, improvement / max(1.0, abs(incumbent)))
                    )
                probability, exploitation = max(
                    potential, key=lambda pair: pair[0] * pair[1], default=(probability, 0.0)
                )
                weighted_quality = probability * exploitation
            exploration = 1.0 / math.sqrt(count + 1.0)
            total = weighted_quality + exploration_weight * exploration
            scores.append(
                RegionScore(
                    region=region,
                    score=total,
                    exploitation=exploitation,
                    exploration=exploration,
                    probability_feasible=probability,
                    observations=count,
                )
            )
        return sorted(scores, key=lambda item: (-item.score, item.region.id))

    def select_regions(
        self,
        records: Sequence[EvaluationRecord],
        count: int,
        rng: random.Random | None = None,
        exploration_probability: float = 0.1,
        **rank_kwargs: Any,
    ) -> tuple[Region, ...]:
        if count < 1:
            raise ValueError("region count must be positive")
        ranked = self.rank_regions(records, **rank_kwargs)
        if not 0 < exploration_probability <= 1:
            raise ValueError("region exploration probability must be in (0,1]")
        generator = rng if rng is not None else random
        chosen = []
        while ranked and len(chosen) < count:
            total = sum(max(0.0, item.score) for item in ranked)
            weights = [
                exploration_probability / len(ranked)
                + (1 - exploration_probability)
                * (max(0.0, item.score) / total if total > 0 else 1 / len(ranked))
                for item in ranked
            ]
            index = generator.choices(range(len(ranked)), weights=weights, k=1)[0]
            chosen.append(ranked.pop(index).region)
        return tuple(chosen)


__all__ = ["ProfilePartitioner", "Region", "RegionScore"]
