"""Stable public API implemented by built-in and third-party runner plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

JsonObject = dict[str, Any]


class RunStatus(StrEnum):
    """Outcome of one real or simulated database evaluation."""

    OK = "ok"
    INVALID = "invalid"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """Engine-neutral workload information passed to runners."""

    dataset: str
    distance: str | None
    top_k: int
    concurrency: int
    vector_size: int | None = None
    filtered: bool = False
    sparse: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """Complete immutable input for one runner evaluation."""

    run_id: str
    engine_id: str
    candidate: Mapping[str, Any]
    rendered_experiment: Mapping[str, Any]
    workload: WorkloadSpec
    seed: int
    timeout_s: float


@dataclass(slots=True)
class Observation:
    """Normalized runner result consumed by the optimization loop."""

    status: RunStatus
    metrics: dict[str, float] = field(default_factory=dict)
    auxiliary: JsonObject = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.OK


@dataclass(frozen=True, slots=True)
class RunnerContext:
    """Construction context supplied to a runner plugin."""

    profile: Any
    settings: Mapping[str, Any]
    execution: Mapping[str, Any]
    artifact_dir: Path


class BaseRunner(ABC):
    """API v1 contract for installed runner plugins.

    A third-party package registers a subclass in the ``eigen.runners`` Python
    entry-point group.  The class must accept :class:`RunnerContext` and return
    one normalized :class:`Observation` per request.
    """

    API_VERSION = "1"
    PLUGIN_ID = "abstract"
    PLUGIN_VERSION = "0"

    def __init__(self, context: RunnerContext) -> None:
        self.context = context

    @abstractmethod
    def evaluate(self, request: EvaluationRequest) -> Observation:
        """Evaluate one rendered candidate."""

    def close(self) -> None:
        """Release runner-owned resources. Implementations may override."""
        return None

    def can_reuse_database_state(self) -> bool:
        """Return whether service restarts may preserve runner-managed state.

        The default is deliberately conservative for third-party runners.  A
        runner opting in must invalidate its cache after every failed or
        interrupted evaluation and must be able to detect incompatible
        workloads itself.
        """

        return False

    def manifest(self) -> JsonObject:
        from eigen import __version__

        return {
            "eigen_version": __version__,
            "plugin_id": self.PLUGIN_ID,
            "plugin_version": self.PLUGIN_VERSION,
            "api_version": self.API_VERSION,
            "class": f"{type(self).__module__}:{type(self).__qualname__}",
        }
