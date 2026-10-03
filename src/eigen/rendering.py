"""Deep-copy rendering from canonical candidates to benchmark experiments."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from eigen.errors import CandidateError, ConfigurationError
from eigen.models import BindingSpec, EngineProfile, JsonScalar
from eigen.search_space import SearchSpace


class ExperimentRenderer:
    """Render complete vector-db-benchmark experiment dictionaries."""

    def __init__(self, profile: EngineProfile) -> None:
        self.profile = profile
        self.search_space = SearchSpace(profile)

    def render(
        self,
        candidate: Mapping[str, Any] | None,
        runtime: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return an independent, complete experiment dictionary."""

        provided_runtime = dict(runtime or {})
        known_runtime = set(self.profile.experiment.runtime_bindings) | set(
            self.profile.experiment.runtime_context
        )
        unknown_runtime = set(provided_runtime) - known_runtime
        if unknown_runtime:
            raise CandidateError(f"unknown runtime values: {sorted(unknown_runtime)}")
        effective_runtime = {
            **copy.deepcopy(self.profile.experiment.runtime_defaults),
            **copy.deepcopy(provided_runtime),
        }

        canonical = self.search_space.canonicalize(candidate, effective_runtime)
        experiment = copy.deepcopy(self.profile.experiment.template)

        consumed: set[str] = set()
        structural = [
            (binding, value)
            for name, value in canonical.items()
            for binding in self.profile.search_space.parameters[name].bindings
            if binding.mode == "container"
        ]
        for binding, value in sorted(structural, key=lambda item: item[0].pointer.count("/")):
            _apply_binding(experiment, binding, value)
        for name, value in canonical.items():
            parameter = self.profile.search_space.parameters[name]
            for binding in parameter.bindings:
                if binding.mode != "container":
                    _apply_binding(experiment, binding, value)
            consumed.add(name)

        missing = set(canonical) - consumed
        if missing:
            raise ConfigurationError(
                f"active candidate parameters were not rendered: {sorted(missing)}"
            )

        for name, bindings in self.profile.experiment.runtime_bindings.items():
            if name not in effective_runtime:
                continue
            for binding in bindings:
                _apply_binding(experiment, binding, effective_runtime[name])

        _validate_rendered_experiment(experiment, self.profile)
        return experiment


def render_experiment(
    profile: EngineProfile,
    candidate: Mapping[str, Any] | None,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`ExperimentRenderer`."""

    return ExperimentRenderer(profile).render(candidate, runtime)


def get_json_pointer(document: Any, pointer: str) -> Any:
    """Read one RFC 6901 JSON Pointer from a JSON-compatible object."""

    current = document
    for token in _pointer_tokens(pointer):
        if isinstance(current, list):
            index = _list_index(token, len(current), allow_end=False)
            current = current[index]
        elif isinstance(current, Mapping):
            if token not in current:
                raise KeyError(pointer)
            current = current[token]
        else:
            raise KeyError(pointer)
    return current


def _apply_binding(document: dict[str, Any], binding: BindingSpec, value: Any) -> None:
    rendered_value = _mapped_value(binding, value)
    if binding.mode == "merge":
        if not isinstance(rendered_value, Mapping):
            raise ConfigurationError(f"merge binding {binding.pointer!r} requires a mapping value")
        try:
            existing = get_json_pointer(document, binding.pointer)
        except KeyError:
            existing = {}
        if not isinstance(existing, Mapping):
            raise ConfigurationError(f"merge target {binding.pointer!r} is not a mapping")
        rendered_value = _deep_merge(existing, rendered_value)
    _set_json_pointer(document, binding.pointer, copy.deepcopy(rendered_value))


def _mapped_value(binding: BindingSpec, value: Any) -> Any:
    if not binding.value_map:
        return value
    key = _value_map_key(value)
    if key not in binding.value_map:
        raise ConfigurationError(
            f"binding {binding.pointer!r} has no value_map entry for {value!r}"
        )
    return binding.value_map[key]


def _value_map_key(value: JsonScalar) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _pointer_tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/") or pointer == "/":
        raise ConfigurationError(f"invalid JSON Pointer {pointer!r}")
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]


def _set_json_pointer(document: Any, pointer: str, value: Any) -> None:
    tokens = _pointer_tokens(pointer)
    current = document
    for index, token in enumerate(tokens[:-1]):
        next_token = tokens[index + 1]
        if isinstance(current, list):
            list_index = _list_index(token, len(current), allow_end=False)
            current = current[list_index]
            continue
        if not isinstance(current, dict):
            raise ConfigurationError(f"cannot traverse {pointer!r} through a non-container")
        if token not in current:
            current[token] = [] if next_token.isdigit() else {}
        current = current[token]

    final = tokens[-1]
    if isinstance(current, list):
        list_index = _list_index(final, len(current), allow_end=False)
        current[list_index] = value
    elif isinstance(current, dict):
        current[final] = value
    else:
        raise ConfigurationError(f"cannot set {pointer!r} on a non-container target")


def _list_index(token: str, length: int, *, allow_end: bool) -> int:
    if not token.isdigit():
        raise ConfigurationError(f"JSON Pointer list token must be numeric, got {token!r}")
    index = int(token)
    limit = length if allow_end else length - 1
    if index < 0 or index > limit:
        raise ConfigurationError(
            f"JSON Pointer list index {index} is outside a list of length {length}"
        )
    return index


def _deep_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(left))
    for key, value in right.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_rendered_experiment(experiment: dict[str, Any], profile: EngineProfile) -> None:
    required_mapping_fields = (
        "connection_params",
        "collection_params",
        "upload_params",
    )
    if not isinstance(experiment.get("name"), str) or not experiment["name"].strip():
        raise ConfigurationError("rendered experiment requires a non-empty name")
    if experiment.get("engine") != profile.adapter.engine:
        raise ConfigurationError("rendered experiment engine does not match the selected profile")
    for field_name in required_mapping_fields:
        if not isinstance(experiment.get(field_name), dict):
            raise ConfigurationError(f"rendered experiment field {field_name!r} must be an object")
    server_params = experiment.get("server_params")
    if server_params is not None and not isinstance(server_params, dict):
        raise ConfigurationError("rendered experiment server_params must be an object")
    search_params = experiment.get("search_params")
    if (
        not isinstance(search_params, list)
        or not search_params
        or not all(isinstance(item, dict) for item in search_params)
    ):
        raise ConfigurationError(
            "rendered experiment search_params must be a non-empty list of objects"
        )
    try:
        json.dumps(experiment, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"rendered experiment is not strict JSON: {error}") from error


__all__ = ["ExperimentRenderer", "get_json_pointer", "render_experiment"]
