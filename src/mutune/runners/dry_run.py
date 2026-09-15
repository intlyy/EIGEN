"""Deterministic runner for validating configuration and optimization flow."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from mutune.api import BaseRunner, EvaluationRequest, Observation, RunStatus
from mutune.errors import ConfigurationError
from mutune.utils import atomic_write_json, fingerprint, safe_name


class DryRunRunner(BaseRunner):
    """Return deterministic synthetic QPS and recall without external effects."""

    PLUGIN_ID = "dry-run"
    PLUGIN_VERSION = "1"

    def __init__(self, context: Any) -> None:
        super().__init__(context)
        metrics_value = context.settings.get(
            "metrics", {"qps": 1000.0, "recall": 0.95, "build_total_time_s": 1.0}
        )
        if not isinstance(metrics_value, Mapping):
            raise ConfigurationError("dry-run settings.metrics must be an object")
        self.metrics = {
            str(name): _finite_float(value, f"dry-run metric {name!r}")
            for name, value in metrics_value.items()
        }
        if self.metrics.get("qps", 0.0) <= 0:
            raise ConfigurationError("dry-run qps must be positive")
        recall = self.metrics.get("recall")
        if recall is None or not 0.0 <= recall <= 1.0:
            raise ConfigurationError("dry-run recall must be in [0, 1]")
        jitter_value = context.settings.get("deterministic_jitter", 0.0)
        self.jitter = _finite_float(jitter_value, "deterministic_jitter")
        if not 0.0 <= self.jitter <= 1.0:
            raise ConfigurationError("deterministic_jitter must be in [0, 1]")

    def evaluate(self, request: EvaluationRequest) -> Observation:
        metrics = dict(self.metrics)
        if self.jitter:
            digest = fingerprint(
                {
                    "engine": request.engine_id,
                    "candidate": dict(request.candidate),
                    "seed": request.seed,
                },
                length=16,
            )
            unit = int(digest, 16) / float(0xFFFFFFFFFFFFFFFF)
            centered = 2.0 * unit - 1.0
            metrics["qps"] = max(1e-12, metrics["qps"] * (1.0 + centered * self.jitter))
            metrics["recall"] = min(
                1.0,
                max(0.0, metrics["recall"] - centered * self.jitter * 0.1),
            )

        artifact_dir = Path(self.context.artifact_dir).expanduser().resolve() / "dry-run"
        artifact_name = (
            f"{safe_name(request.run_id, 60)}-"
            f"{fingerprint({'candidate': dict(request.candidate), 'seed': request.seed})}.json"
        )
        artifact_path = artifact_dir / artifact_name
        atomic_write_json(
            artifact_path,
            {
                "runner": self.manifest(),
                "run_id": request.run_id,
                "engine_id": request.engine_id,
                "candidate": dict(request.candidate),
                "rendered_experiment": dict(request.rendered_experiment),
                "workload": {
                    "dataset": request.workload.dataset,
                    "distance": request.workload.distance,
                    "top_k": request.workload.top_k,
                    "concurrency": request.workload.concurrency,
                    "vector_size": request.workload.vector_size,
                    "filtered": request.workload.filtered,
                    "sparse": request.workload.sparse,
                },
                "seed": request.seed,
                "metrics": metrics,
            },
        )
        return Observation(
            status=RunStatus.OK,
            metrics=metrics,
            auxiliary={"simulated": True},
            artifacts=[str(artifact_path)],
        )


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{label} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ConfigurationError(f"{label} must be finite")
    return parsed


__all__ = ["DryRunRunner"]
