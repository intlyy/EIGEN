"""Canonical validation, activation, sampling, and effect tracking."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

from eigen.errors import CandidateError
from eigen.models import (
    ConstraintSpec,
    EngineProfile,
    JsonScalar,
    Operand,
    ParameterSpec,
    Predicate,
    SearchSpaceSpec,
)


class SearchSpace:
    """Executable view of a declarative :class:`SearchSpaceSpec`."""

    def __init__(
        self,
        profile_or_spec: EngineProfile | SearchSpaceSpec,
        runtime_defaults: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(profile_or_spec, EngineProfile):
            self.spec = profile_or_spec.search_space
            defaults = profile_or_spec.experiment.runtime_defaults
            self._runtime_keys: frozenset[str] | None = frozenset(
                set(profile_or_spec.experiment.runtime_bindings)
                | set(profile_or_spec.experiment.runtime_context)
            )
        else:
            self.spec = profile_or_spec
            defaults = {}
            self._runtime_keys = None
        self.runtime_defaults = {**defaults, **dict(runtime_defaults or {})}

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.spec.parameters)

    def canonicalize(
        self,
        candidate: Mapping[str, Any] | None,
        runtime: Mapping[str, Any] | None = None,
        *,
        reject_inactive: bool = True,
    ) -> dict[str, JsonScalar]:
        """Validate a partial candidate, fill defaults, and remove inactive keys."""

        if candidate is None:
            candidate = {}
        if not isinstance(candidate, Mapping):
            raise CandidateError("candidate must be a mapping")

        supplied = set(candidate)
        unknown = supplied - set(self.spec.parameters)
        if unknown:
            raise CandidateError(f"unknown candidate parameters: {sorted(unknown)}")

        effective_runtime = self._effective_runtime(runtime)
        all_values = {
            name: _normalize_value(name, candidate.get(name, spec.default), spec)
            for name, spec in self.spec.parameters.items()
        }

        active_cache: dict[str, bool] = {}

        def is_active(name: str) -> bool:
            cached = active_cache.get(name)
            if cached is not None:
                return cached
            spec = self.spec.parameters[name]
            active = all(
                is_active(predicate.parameter)
                and _predicate_matches(predicate, all_values[predicate.parameter])
                for predicate in spec.active_if
            )
            active_cache[name] = active
            return active

        active_names = {name for name in self.spec.parameters if is_active(name)}
        explicitly_inactive = supplied - active_names
        if reject_inactive and explicitly_inactive:
            raise CandidateError(
                f"candidate explicitly sets inactive parameters: {sorted(explicitly_inactive)}"
            )

        canonical = {
            name: all_values[name] for name in self.spec.parameters if name in active_names
        }
        self._validate_constraints(canonical, all_values, active_names, effective_runtime)
        return canonical

    def normalize(
        self,
        candidate: Mapping[str, Any] | None,
        runtime: Mapping[str, Any] | None = None,
        *,
        reject_inactive: bool = True,
    ) -> dict[str, JsonScalar]:
        """Alias for :meth:`canonicalize`."""

        return self.canonicalize(candidate, runtime, reject_inactive=reject_inactive)

    def validate(
        self,
        candidate: Mapping[str, Any] | None,
        runtime: Mapping[str, Any] | None = None,
        *,
        reject_inactive: bool = True,
    ) -> dict[str, JsonScalar]:
        """Validate and return the canonical candidate."""

        return self.canonicalize(candidate, runtime, reject_inactive=reject_inactive)

    def sample(
        self,
        rng: random.Random,
        runtime: Mapping[str, Any] | None = None,
        *,
        max_attempts: int = 1_000,
        fixed: Mapping[str, Any] | None = None,
    ) -> dict[str, JsonScalar]:
        """Sample a valid canonical candidate with deterministic caller-owned RNG."""

        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        for _ in range(max_attempts):
            raw = {name: _sample_value(spec, rng) for name, spec in self.spec.parameters.items()}
            raw.update(fixed or {})
            try:
                return self.canonicalize(raw, runtime, reject_inactive=False)
            except CandidateError:
                continue
        raise CandidateError(f"could not sample a valid candidate after {max_attempts} attempts")

    def active_parameters(
        self,
        candidate: Mapping[str, Any] | None,
        runtime: Mapping[str, Any] | None = None,
    ) -> set[str]:
        """Return the active canonical parameter names for a candidate."""

        return set(self.canonicalize(candidate, runtime, reject_inactive=False))

    def effects_between(
        self,
        left: Mapping[str, Any] | None,
        right: Mapping[str, Any] | None,
        runtime: Mapping[str, Any] | None = None,
    ) -> frozenset[str]:
        """Return lifecycle effects caused by changing between two candidates."""

        left_point = self.canonicalize(left, runtime, reject_inactive=False)
        right_point = self.canonicalize(right, runtime, reject_inactive=False)
        changed = {
            name
            for name in set(left_point) | set(right_point)
            if not _same_value(left_point.get(name), right_point.get(name))
        }
        return frozenset(self.spec.parameters[name].effect for name in changed)

    def _effective_runtime(self, runtime: Mapping[str, Any] | None) -> dict[str, Any]:
        provided = dict(runtime or {})
        if self._runtime_keys is not None:
            unknown = set(provided) - self._runtime_keys
            if unknown:
                raise CandidateError(f"unknown runtime values: {sorted(unknown)}")
        return {**self.runtime_defaults, **provided}

    def _validate_constraints(
        self,
        canonical: Mapping[str, JsonScalar],
        all_values: Mapping[str, JsonScalar],
        active_names: set[str],
        runtime: Mapping[str, Any],
    ) -> None:
        for constraint in self.spec.constraints:
            if not _constraint_is_enabled(constraint, all_values, active_names):
                continue
            left = _resolve_operand(constraint.left, canonical, runtime)
            right = _resolve_operand(constraint.right, canonical, runtime)
            try:
                satisfied = _compare(left, constraint.op, right)
            except TypeError as error:
                raise CandidateError(
                    f"constraint {constraint.id!r} compares incompatible values: "
                    f"{left!r} and {right!r}"
                ) from error
            if not satisfied:
                raise CandidateError(
                    f"constraint {constraint.id!r} failed: {constraint.message} "
                    f"(left={left!r}, right={right!r})"
                )


def _normalize_value(name: str, value: Any, spec: ParameterSpec) -> JsonScalar:
    if spec.kind == "integer":
        if type(value) is not int:
            raise CandidateError(f"parameter {name!r} must be an integer")
        low, high = spec.bounds or (0, 0)
        if not int(low) <= value <= int(high):
            raise CandidateError(f"parameter {name!r}={value} is outside [{int(low)}, {int(high)}]")
        step = int(spec.step or 1)
        if (value - int(low)) % step:
            raise CandidateError(f"parameter {name!r} does not align to step {step}")
        return value

    if spec.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CandidateError(f"parameter {name!r} must be numeric")
        normalized = float(value)
        low, high = spec.bounds or (0.0, 0.0)
        if not float(low) <= normalized <= float(high):
            raise CandidateError(f"parameter {name!r}={normalized} is outside [{low}, {high}]")
        if spec.step is not None:
            quotient = (normalized - float(low)) / float(spec.step)
            if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
                raise CandidateError(f"parameter {name!r} does not align to step {spec.step}")
        return normalized

    if spec.kind == "categorical":
        for choice in spec.choices or []:
            if _same_value(value, choice):
                return choice
        raise CandidateError(f"parameter {name!r}={value!r} is not one of {spec.choices!r}")

    if type(value) is not bool:
        raise CandidateError(f"parameter {name!r} must be a boolean")
    return value


def _predicate_matches(predicate: Predicate, value: JsonScalar) -> bool:
    if predicate.op == "eq":
        return _same_value(value, predicate.value)
    if predicate.op == "ne":
        return not _same_value(value, predicate.value)
    choices = predicate.value
    contains = any(_same_value(value, choice) for choice in choices)
    return contains if predicate.op == "in" else not contains


def _constraint_is_enabled(
    constraint: ConstraintSpec,
    values: Mapping[str, JsonScalar],
    active_names: set[str],
) -> bool:
    return all(
        predicate.parameter in active_names
        and _predicate_matches(predicate, values[predicate.parameter])
        for predicate in constraint.when
    )


def _resolve_operand(
    operand: Operand,
    candidate: Mapping[str, JsonScalar],
    runtime: Mapping[str, Any],
) -> Any:
    if operand.source == "literal":
        return operand.value
    source = candidate if operand.source == "candidate" else runtime
    if operand.key not in source:
        raise CandidateError(f"constraint operand {operand.source}.{operand.key} is unavailable")
    return source[operand.key]


def _compare(left: Any, op: str, right: Any) -> bool:
    if op == "lt":
        return left < right
    if op == "le":
        return left <= right
    if op == "eq":
        return _same_value(left, right)
    if op == "ne":
        return not _same_value(left, right)
    if op == "ge":
        return left >= right
    if op == "divides":
        if type(left) is not int or type(right) is not int or left == 0:
            raise TypeError("divides requires a non-zero integer divisor and integer value")
        return right % left == 0
    return left > right


def _sample_value(spec: ParameterSpec, rng: random.Random) -> JsonScalar:
    if spec.kind == "integer":
        low, high = (int(value) for value in (spec.bounds or [0, 0]))
        step = int(spec.step or 1)
        if spec.log:
            raw = math.exp(rng.uniform(math.log(low), math.log(high)))
            index = round((raw - low) / step)
            return max(low, min(high, low + index * step))
        return low + rng.randrange(((high - low) // step) + 1) * step

    if spec.kind == "float":
        low, high = (float(value) for value in (spec.bounds or [0.0, 0.0]))
        if spec.log:
            value = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:
            value = rng.uniform(low, high)
        if spec.step is not None:
            value = low + round((value - low) / float(spec.step)) * float(spec.step)
            value = max(low, min(high, value))
        return value

    if spec.kind == "categorical":
        return rng.choice(spec.choices or [])

    return bool(rng.getrandbits(1))


def _same_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


__all__ = ["SearchSpace"]
