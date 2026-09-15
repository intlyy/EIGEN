"""Strict, declarative models for engine profiles and search spaces."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JsonScalar = str | int | float | bool | None
ParameterEffect = Literal["search", "index", "collection", "server"]

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PARAMETER_NAME = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")


class StrictModel(BaseModel):
    """Base model that rejects misspelled or future-unknown fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class AdapterRef(StrictModel):
    """Installed runner plugin and the engine name understood by that runner."""

    plugin: str
    engine: str
    api_version: Literal["1"] = "1"
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plugin", "engine")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"invalid identifier: {value!r}")
        return value


class Capabilities(StrictModel):
    """Capabilities exercised by this profile, not every feature of the engine."""

    vector_kinds: list[Literal["dense", "sparse"]]
    distances: list[Literal["l2", "cosine", "dot"]]
    filters: list[Literal["exact", "range", "geo"]] = Field(default_factory=list)
    indexes: list[str]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_nonempty_unique_values(self) -> "Capabilities":
        for field_name in ("vector_kinds", "distances", "filters", "indexes"):
            values = getattr(self, field_name)
            if field_name != "filters" and not values:
                raise ValueError(f"capabilities.{field_name} cannot be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"capabilities.{field_name} contains duplicates")
        return self


class BindingSpec(StrictModel):
    """Map one canonical/runtime value to a JSON Pointer in an experiment."""

    pointer: str
    mode: Literal["set", "merge", "container"] = "set"
    # Mapping values may be complete JSON fragments.  This is needed for
    # engines whose index selector expands to a native object (for example a
    # Qdrant quantization configuration), rather than to a single scalar.
    value_map: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_container(self) -> BindingSpec:
        if self.mode == "container" and (
            not self.value_map
            or any(
                value is not None and not isinstance(value, dict)
                for value in self.value_map.values()
            )
        ):
            raise ValueError("container bindings require explicit object/null variants")
        return self

    @field_validator("pointer")
    @classmethod
    def validate_pointer(cls, value: str) -> str:
        if not value.startswith("/") or value == "/" or "//" in value:
            raise ValueError("pointer must be a non-empty JSON Pointer")
        return value

    @field_validator("value_map")
    @classmethod
    def validate_value_map(cls, value: dict[str, Any]) -> dict[str, Any]:
        for key, mapped in value.items():
            if not key:
                raise ValueError("value_map keys cannot be empty")
            _validate_json_value(mapped, path=f"value_map[{key!r}]")
        return value


class Predicate(StrictModel):
    """An activation predicate; predicates in a list are combined with AND."""

    parameter: str
    op: Literal["eq", "ne", "in", "not_in"]
    value: JsonScalar | list[JsonScalar]

    @field_validator("parameter")
    @classmethod
    def validate_parameter_name(cls, value: str) -> str:
        if not _PARAMETER_NAME.fullmatch(value):
            raise ValueError(f"invalid parameter name: {value!r}")
        return value

    @model_validator(mode="after")
    def validate_operator_value(self) -> "Predicate":
        if self.op in {"in", "not_in"} and not isinstance(self.value, list):
            raise ValueError(f"predicate {self.op!r} requires a list value")
        if self.op in {"eq", "ne"} and isinstance(self.value, (list, dict)):
            raise ValueError(f"predicate {self.op!r} requires a scalar value")
        return self


class ParameterSpec(StrictModel):
    """One engine-neutral optimization parameter."""

    kind: Literal["integer", "float", "categorical", "boolean"]
    default: JsonScalar
    bounds: list[int | float] | None = None
    choices: list[JsonScalar] | None = None
    step: int | float | None = None
    log: bool = False
    effect: ParameterEffect
    active_if: list[Predicate] = Field(default_factory=list)
    bindings: list[BindingSpec]
    description: str = ""

    @model_validator(mode="after")
    def validate_domain(self) -> "ParameterSpec":
        if not self.bindings:
            raise ValueError("every tunable parameter must have at least one binding")

        if self.kind in {"integer", "float"}:
            if self.bounds is None or len(self.bounds) != 2:
                raise ValueError("numeric parameters require two bounds")
            if self.choices is not None:
                raise ValueError("numeric parameters cannot define choices")
            low, high = self.bounds
            if isinstance(low, bool) or isinstance(high, bool) or low >= high:
                raise ValueError("numeric bounds must satisfy low < high")
            if isinstance(self.default, bool) or not isinstance(self.default, (int, float)):
                raise ValueError("numeric default must be an int or float")
            if not float(low) <= float(self.default) <= float(high):
                raise ValueError("numeric default is outside bounds")
            if self.kind == "integer":
                values = (low, high, self.default)
                if not all(float(value).is_integer() for value in values):
                    raise ValueError("integer parameter domain must be integral")
                if self.step is not None and not float(self.step).is_integer():
                    raise ValueError("integer parameter step must be integral")
            if self.step is not None and self.step <= 0:
                raise ValueError("step must be positive")
            if self.log and low <= 0:
                raise ValueError("log-scaled bounds must be positive")
        elif self.kind == "categorical":
            if not self.choices:
                raise ValueError("categorical parameters require choices")
            if self.bounds is not None or self.step is not None or self.log:
                raise ValueError("categorical parameters cannot define a numeric domain")
            if not any(_same_scalar(self.default, choice) for choice in self.choices):
                raise ValueError("categorical default is not one of choices")
            if len({_typed_scalar_key(choice) for choice in self.choices}) != len(self.choices):
                raise ValueError("categorical choices contain duplicates")
        else:
            if type(self.default) is not bool:
                raise ValueError("boolean default must be a bool")
            if self.bounds is not None or self.choices is not None or self.step is not None:
                raise ValueError("boolean parameters cannot define bounds, choices, or step")
            if self.log:
                raise ValueError("boolean parameters cannot use log sampling")
        return self


class Operand(StrictModel):
    """One side of a declarative relational constraint."""

    source: Literal["candidate", "runtime", "literal"]
    key: str | None = None
    value: JsonScalar = None

    @model_validator(mode="after")
    def validate_source(self) -> "Operand":
        if self.source in {"candidate", "runtime"} and not self.key:
            raise ValueError("candidate/runtime operands require key")
        if self.source == "literal" and self.key is not None:
            raise ValueError("literal operands cannot define key")
        return self


class ConstraintSpec(StrictModel):
    """A safe relation between candidate, runtime, or literal values."""

    id: str
    left: Operand
    op: Literal["lt", "le", "eq", "ne", "ge", "gt", "divides"]
    right: Operand
    when: list[Predicate] = Field(default_factory=list)
    message: str


class SearchSpaceSpec(StrictModel):
    """Canonical parameter space shared by one or more engine profiles."""

    domain: str
    partition_by: list[str] = Field(default_factory=list)
    parameters: dict[str, ParameterSpec]
    constraints: list[ConstraintSpec] = Field(default_factory=list)

    @field_validator("parameters")
    @classmethod
    def validate_parameter_names(
        cls, parameters: dict[str, ParameterSpec]
    ) -> dict[str, ParameterSpec]:
        if not parameters:
            raise ValueError("search space cannot be empty")
        invalid = [name for name in parameters if not _PARAMETER_NAME.fullmatch(name)]
        if invalid:
            raise ValueError(f"invalid parameter names: {invalid}")
        return parameters

    @model_validator(mode="after")
    def validate_parameter_references(self) -> "SearchSpaceSpec":
        parameter_names = set(self.parameters)
        if len(self.partition_by) != len(set(self.partition_by)):
            raise ValueError("search_space.partition_by contains duplicates")
        for name in self.partition_by:
            if name not in parameter_names:
                raise ValueError(f"search_space.partition_by references unknown parameter {name!r}")
            if self.parameters[name].kind not in {"categorical", "boolean"}:
                raise ValueError(f"partition parameter {name!r} must be categorical or boolean")
        for name, parameter in self.parameters.items():
            for predicate in parameter.active_if:
                if predicate.parameter not in parameter_names:
                    raise ValueError(
                        f"parameter {name!r} references unknown parameter {predicate.parameter!r}"
                    )
        for constraint in self.constraints:
            for predicate in constraint.when:
                if predicate.parameter not in parameter_names:
                    raise ValueError(
                        f"constraint {constraint.id!r} references unknown parameter "
                        f"{predicate.parameter!r}"
                    )
            for operand in (constraint.left, constraint.right):
                if operand.source == "candidate" and operand.key not in parameter_names:
                    raise ValueError(
                        f"constraint {constraint.id!r} references unknown parameter {operand.key!r}"
                    )
        _validate_activation_graph(self.parameters)
        return self


class ExperimentSpec(StrictModel):
    """Benchmark experiment template and declarative input/output mappings."""

    template: dict[str, Any]
    runtime_defaults: dict[str, Any] = Field(default_factory=dict)
    runtime_bindings: dict[str, list[BindingSpec]] = Field(default_factory=dict)
    # Values available to constraints but intentionally not written into the
    # rendered benchmark JSON (for example the dataset vector dimension).
    runtime_context: list[str] = Field(default_factory=list)
    metric_pointers: dict[str, str]

    @field_validator("metric_pointers")
    @classmethod
    def validate_metric_pointers(cls, pointers: dict[str, str]) -> dict[str, str]:
        if not pointers:
            raise ValueError("at least one metric pointer is required")
        for name, pointer in pointers.items():
            if not name or not pointer.startswith("/"):
                raise ValueError("metric mappings require names and JSON Pointers")
        return pointers

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> "ExperimentSpec":
        if len(self.runtime_context) != len(set(self.runtime_context)):
            raise ValueError("experiment.runtime_context contains duplicates")
        overlap = set(self.runtime_context) & set(self.runtime_bindings)
        if overlap:
            raise ValueError(
                f"runtime values cannot be both bound and context-only: {sorted(overlap)}"
            )
        declared = set(self.runtime_bindings) | set(self.runtime_context)
        unknown_defaults = set(self.runtime_defaults) - declared
        if unknown_defaults:
            raise ValueError(f"runtime defaults lack bindings: {sorted(unknown_defaults)}")
        for name, bindings in self.runtime_bindings.items():
            if not bindings:
                raise ValueError(f"runtime binding {name!r} cannot be empty")
        return self


class LifecycleSpec(StrictModel):
    """How parameter effects change reusable benchmark state."""

    reset: Literal["adapter"] = "adapter"
    readiness: Literal["adapter"] = "adapter"
    rebuild_effects: list[ParameterEffect] = Field(default_factory=lambda: ["collection", "index"])
    restart_effects: list[ParameterEffect] = Field(default_factory=lambda: ["server"])
    reusable_effects: list[ParameterEffect] = Field(default_factory=lambda: ["search"])
    supports_search_batching: bool = True


class EngineProfile(StrictModel):
    """Complete declarative description of one tunable engine/index profile."""

    schema_version: Literal[1]
    id: str
    display_name: str
    adapter: AdapterRef
    capabilities: Capabilities
    experiment: ExperimentSpec
    search_space: SearchSpaceSpec
    lifecycle: LifecycleSpec = Field(default_factory=LifecycleSpec)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError(f"invalid profile id: {value!r}")
        return value

    @model_validator(mode="after")
    def validate_cross_references(self) -> "EngineProfile":
        if self.experiment.template.get("engine") != self.adapter.engine:
            raise ValueError("experiment.template.engine must equal adapter.engine")

        parameters = self.search_space.parameters
        parameter_names = set(parameters)
        pointer_owners: dict[str, tuple[str, str]] = {}

        for name, parameter in parameters.items():
            for predicate in parameter.active_if:
                if predicate.parameter not in parameter_names:
                    raise ValueError(
                        f"parameter {name!r} references unknown parameter {predicate.parameter!r}"
                    )
            for binding in parameter.bindings:
                if binding.pointer in {"/name", "/engine"}:
                    raise ValueError("candidate parameters cannot overwrite name or engine")
                if parameter.effect == "server" and not binding.pointer.startswith(
                    "/server_params/"
                ):
                    raise ValueError(f"server parameter {name!r} must bind below /server_params")
                if parameter.effect != "server" and binding.pointer.startswith("/server_params/"):
                    raise ValueError(
                        f"parameter {name!r} binds below /server_params and must have "
                        "effect='server'"
                    )
                if binding.value_map:
                    if parameter.kind not in {"categorical", "boolean"}:
                        raise ValueError(
                            f"parameter {name!r} uses value_map but is not categorical or boolean"
                        )
                    values = (
                        list(parameter.choices or [])
                        if parameter.kind == "categorical"
                        else [False, True]
                    )
                    expected_keys = {_binding_value_key(value) for value in values}
                    actual_keys = set(binding.value_map)
                    if actual_keys != expected_keys:
                        raise ValueError(
                            f"parameter {name!r} value_map keys must exactly match its "
                            f"domain; missing={sorted(expected_keys - actual_keys)}, "
                            f"extra={sorted(actual_keys - expected_keys)}"
                        )
                _claim_pointer(pointer_owners, binding.pointer, f"candidate:{name}", binding.mode)

        for runtime_name, bindings in self.experiment.runtime_bindings.items():
            for binding in bindings:
                _claim_pointer(
                    pointer_owners,
                    binding.pointer,
                    f"runtime:{runtime_name}",
                    binding.mode,
                )

        for constraint in self.search_space.constraints:
            for predicate in constraint.when:
                if predicate.parameter not in parameter_names:
                    raise ValueError(
                        f"constraint {constraint.id!r} references unknown parameter "
                        f"{predicate.parameter!r}"
                    )
            for operand in (constraint.left, constraint.right):
                if operand.source == "candidate" and operand.key not in parameter_names:
                    raise ValueError(
                        f"constraint {constraint.id!r} references unknown parameter {operand.key!r}"
                    )
                if operand.source == "runtime" and operand.key not in (
                    set(self.experiment.runtime_bindings) | set(self.experiment.runtime_context)
                ):
                    raise ValueError(
                        f"constraint {constraint.id!r} references unknown runtime key "
                        f"{operand.key!r}"
                    )

        _validate_activation_graph(parameters)
        return self


def _validate_activation_graph(parameters: dict[str, ParameterSpec]) -> None:
    graph = {
        name: {predicate.parameter for predicate in parameter.active_if}
        for name, parameter in parameters.items()
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, path: list[str]) -> None:
        if name in visiting:
            start = path.index(name)
            cycle = path[start:] + [name]
            raise ValueError(f"activation dependency cycle: {' -> '.join(cycle)}")
        if name in visited:
            return
        visiting.add(name)
        path.append(name)
        for dependency in graph[name]:
            visit(dependency, path)
        path.pop()
        visiting.remove(name)
        visited.add(name)

    for parameter_name in graph:
        visit(parameter_name, [])


def _claim_pointer(
    owners: dict[str, tuple[str, str]], pointer: str, owner: str, mode: str = "set"
) -> None:
    for claimed_pointer, (claimed_owner, claimed_mode) in owners.items():
        # Explicit structural variants are initialized before leaf bindings.
        # Same-pointer writes and ordinary parent/child overwrites still fail.
        if claimed_mode == "container" and pointer.startswith(f"{claimed_pointer}/"):
            continue
        if mode == "container" and claimed_pointer.startswith(f"{pointer}/"):
            continue
        overlaps = (
            pointer == claimed_pointer
            or pointer.startswith(f"{claimed_pointer}/")
            or claimed_pointer.startswith(f"{pointer}/")
        )
        if overlaps:
            raise ValueError(
                f"binding collision between {claimed_pointer!r} ({claimed_owner}) "
                f"and {pointer!r} ({owner})"
            )
    owners[pointer] = (owner, mode)


def _same_scalar(left: JsonScalar, right: JsonScalar) -> bool:
    return type(left) is type(right) and left == right


def _typed_scalar_key(value: JsonScalar) -> tuple[type[JsonScalar], JsonScalar]:
    return type(value), value


def _binding_value_key(value: JsonScalar) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_json_value(value: Any, *, path: str) -> None:
    """Reject non-JSON and non-finite values in declarative binding fragments."""

    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite JSON")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} object keys must be strings")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must contain strict JSON values") from error
    raise ValueError(f"{path} must contain strict JSON values")
