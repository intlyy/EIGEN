"""Candidate proposers with canonical validation and bounded retries."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Protocol, Sequence

from mutune.errors import CandidateError
from mutune.llm import CompletionClient, CompletionResult, LLMError, parse_json_response

from .history import EvaluationRecord, candidate_key
from .partitioning import Region


class CandidateProposer(Protocol):
    def propose(
        self,
        count: int,
        *,
        region: Region,
        history: Sequence[EvaluationRecord],
        excluded_keys: set[str] | frozenset[str],
        deadline: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``count`` unseen canonical candidates in ``region``."""


def _canonicalize(
    search_space: Any,
    candidate: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    reject_inactive: bool,
) -> dict[str, Any]:
    normalized = search_space.canonicalize(
        dict(candidate),
        runtime=runtime,
        reject_inactive=reject_inactive,
    )
    if not isinstance(normalized, Mapping):
        raise TypeError("SearchSpace.canonicalize must return a mapping")
    return dict(normalized)


class RandomProposer:
    """Profile-aware random sampling used for initialization and fallback."""

    def __init__(
        self,
        search_space: Any,
        *,
        rng: random.Random,
        runtime: Mapping[str, Any] | None = None,
        attempt_multiplier: int = 50,
    ) -> None:
        if attempt_multiplier < 1:
            raise ValueError("attempt_multiplier must be positive")
        self.search_space = search_space
        self.rng = rng
        self.runtime = dict(runtime or {})
        self.attempt_multiplier = attempt_multiplier

    def propose(
        self,
        count: int,
        *,
        region: Region,
        history: Sequence[EvaluationRecord] = (),
        excluded_keys: set[str] | frozenset[str] = frozenset(),
        deadline: float | None = None,
    ) -> list[dict[str, Any]]:
        del history
        if count < 1:
            return []
        excluded = set(excluded_keys)
        proposed: list[dict[str, Any]] = []
        local_keys: set[str] = set()
        max_attempts = max(100, count * self.attempt_multiplier)

        for _ in range(max_attempts):
            if len(proposed) >= count:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                raw = self.search_space.sample(self.rng, runtime=self.runtime, fixed=region.fixed)
                raw = dict(raw)
                raw.update(region.fixed)
                candidate = _canonicalize(
                    self.search_space,
                    raw,
                    self.runtime,
                    # ``sample`` may have selected another conditional branch
                    # before the region selector is fixed above.  Prune those
                    # now-inactive values instead of discarding a valid draw.
                    reject_inactive=False,
                )
            except (CandidateError, KeyError, TypeError, ValueError):
                continue
            if not region.matches(candidate):
                continue
            key = candidate_key(candidate)
            if key in excluded or key in local_keys:
                continue
            local_keys.add(key)
            proposed.append(candidate)
        return proposed


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _search_space_description(search_space: Any) -> Any:
    for value in (
        getattr(search_space, "spec", None),
        getattr(search_space, "search_space", None),
    ):
        if value is not None:
            return _jsonable(value)
    profile = getattr(search_space, "profile", None)
    if profile is not None:
        return _jsonable(getattr(profile, "search_space", profile))
    return "Profile-backed conditional search space"


class OpenAICompatibleProposer:
    """Generate profile-valid candidates through a bounded completion contract."""

    def __init__(
        self,
        search_space: Any,
        client: CompletionClient,
        *,
        runtime: Mapping[str, Any] | None = None,
        objective_metric: str,
        constraint_metric: str,
        constraint_threshold: float,
        max_attempts: int = 3,
        history_limit: int = 50,
        task: Mapping[str, Any] | None = None,
        objectives: Sequence[Any] = (),
        guidance_objectives: Sequence[Any] = (),
        constraints: Sequence[Any] = (),
        monotonic=time.monotonic,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self.search_space = search_space
        self.client = client
        self.runtime = dict(runtime or {})
        self.objective_metric = objective_metric
        self.constraint_metric = constraint_metric
        self.constraint_threshold = constraint_threshold
        self.max_attempts = max_attempts
        self.history_limit = history_limit
        self.task = dict(task or {})
        self.objectives = list(objectives)
        self.guidance_objectives = list(guidance_objectives)
        self.constraints = list(constraints)
        self._monotonic = monotonic

    def _prompt(
        self,
        count: int,
        region: Region,
        history: Sequence[EvaluationRecord],
    ) -> str:
        observations = [
            {"candidate": record.candidate, "metrics": record.metrics}
            for record in history[-self.history_limit :]
            if record.ok
            and self.objective_metric in record.metrics
            and self.constraint_metric in record.metrics
        ]
        contract = {
            "task": "propose unseen vector-database configurations",
            "workload": self.task,
            "runtime": {
                key: value for key, value in self.runtime.items() if key != "connection_params"
            },
            "efficiency_objectives": _jsonable(self.objectives),
            "archive_coordinates": _jsonable(self.guidance_objectives),
            "all_constraints": _jsonable(self.constraints),
            "number_of_candidates": count,
            "objective": {"metric": self.objective_metric, "direction": "maximize"},
            "constraint": {
                "metric": self.constraint_metric,
                "operator": ">=",
                "threshold": self.constraint_threshold,
            },
            "region": {
                "id": region.id,
                "fixed": region.fixed,
                "active_parameters": list(region.active_parameters),
            },
            "search_space": _search_space_description(self.search_space),
            "observations": observations,
            "output_contract": (
                "Return only a JSON array of configuration objects. Use canonical "
                "parameter names and concrete JSON values; do not include commentary."
            ),
        }
        return json.dumps(contract, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _candidate_items(payload: Any) -> list[Mapping[str, Any]]:
        if isinstance(payload, Mapping):
            for key in ("candidates", "configurations", "points"):
                value = payload.get(key)
                if isinstance(value, list):
                    payload = value
                    break
            else:
                payload = [payload]
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, Mapping)]

    def propose(
        self,
        count: int,
        *,
        region: Region,
        history: Sequence[EvaluationRecord],
        excluded_keys: set[str] | frozenset[str],
        deadline: float | None = None,
    ) -> list[dict[str, Any]]:
        if count < 1:
            return []
        excluded = set(excluded_keys)
        local_keys: set[str] = set()
        proposed: list[dict[str, Any]] = []
        last_error: BaseException | None = None

        for _ in range(self.max_attempts):
            if len(proposed) >= count:
                break
            if deadline is not None and self._monotonic() >= deadline:
                break
            prompt = self._prompt(count - len(proposed), region, history)
            try:
                response = self.client.complete(prompt, deadline=deadline)
                content = (
                    response.content if isinstance(response, CompletionResult) else str(response)
                )
                payload = parse_json_response(content)
            except (LLMError, TypeError, ValueError) as error:
                last_error = error
                continue

            for raw in self._candidate_items(payload):
                try:
                    if any(key in raw and raw[key] != value for key, value in region.fixed.items()):
                        raise CandidateError("LLM candidate contradicts its selected region")
                    merged = dict(raw)
                    merged.update(region.fixed)
                    candidate = _canonicalize(
                        self.search_space,
                        merged,
                        self.runtime,
                        reject_inactive=True,
                    )
                except (CandidateError, KeyError, TypeError, ValueError) as error:
                    last_error = error
                    continue
                if not region.matches(candidate):
                    continue
                key = candidate_key(candidate)
                if key in excluded or key in local_keys:
                    continue
                local_keys.add(key)
                proposed.append(candidate)
                if len(proposed) >= count:
                    break

        # Returning a partial batch lets HybridProposer fill it deterministically.
        # Pure-LLM callers can decide whether a partial result is acceptable.
        del last_error
        return proposed


class HybridProposer:
    """Prefer LLM proposals and fill short batches with profile-random points."""

    def __init__(self, primary: CandidateProposer, fallback: RandomProposer) -> None:
        self.primary = primary
        self.fallback = fallback

    def propose(
        self,
        count: int,
        *,
        region: Region,
        history: Sequence[EvaluationRecord],
        excluded_keys: set[str] | frozenset[str],
        deadline: float | None = None,
    ) -> list[dict[str, Any]]:
        primary = self.primary.propose(
            count,
            region=region,
            history=history,
            excluded_keys=excluded_keys,
            deadline=deadline,
        )
        if len(primary) >= count:
            return primary[:count]
        combined_excluded = set(excluded_keys)
        combined_excluded.update(candidate_key(candidate) for candidate in primary)
        fallback = self.fallback.propose(
            count - len(primary),
            region=region,
            history=history,
            excluded_keys=combined_excluded,
            deadline=deadline,
        )
        return primary + fallback


__all__ = [
    "CandidateProposer",
    "HybridProposer",
    "OpenAICompatibleProposer",
    "RandomProposer",
]
