"""Shared lifecycle composition for local tuning and fixed measurements."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from eigen.api import EvaluationRequest, RunnerContext, WorkloadSpec
from eigen.config import LoadedProject
from eigen.lifecycle import create_lifecycle
from eigen.llm import OpenAICompatibleClient
from eigen.plugins import create_runner
from eigen.profiles import validate_workload
from eigen.rendering import ExperimentRenderer
from eigen.search_space import SearchSpace
from eigen.tuning.tuner import Tuner
from eigen.utils import atomic_write_json, safe_name


class ProjectSession:
    """One exclusive database endpoint; never share a session between workers."""

    def __init__(self, project: LoadedProject) -> None:
        self.project = project
        cfg = project.config
        self.execution = cfg.execution.model_dump(mode="json")
        self.workload = WorkloadSpec(
            cfg.execution.dataset,
            cfg.execution.distance,
            cfg.execution.top_k,
            cfg.execution.search_parallel,
            cfg.execution.vector_size,
            cfg.execution.filtered,
            cfg.execution.sparse,
        )
        validate_workload(project.profile, self.workload)
        self.lifecycle = create_lifecycle(
            cfg.lifecycle.model_dump(mode="json"),
            workspace_root=project.source_path.parent,
            artifact_dir=project.artifact_dir / "lifecycle",
            default_endpoint=cfg.execution.host,
            default_project_name="eigen-"
            + safe_name(cfg.experiment_name, 50).lower().replace(".", "-"),
        )
        self.execution["host"] = self.lifecycle.endpoint
        self.runner = create_runner(
            cfg.runner.plugin or project.profile.adapter.plugin,
            RunnerContext(
                project.profile,
                {**project.profile.adapter.options, **cfg.runner.settings},
                self.execution,
                project.artifact_dir,
            ),
        )
        self.started = False
        self.runner_owned_by_tuner = False
        available = {
            "experiment_name": cfg.experiment_name,
            "connection_params": cfg.execution.connection_params,
            "upload_parallel": cfg.execution.upload_parallel,
            "search_parallel": cfg.execution.search_parallel,
            "top_k": cfg.execution.top_k,
            "batch_size": cfg.execution.batch_size,
            "vector_size": cfg.execution.vector_size,
        }
        declared = set(project.profile.experiment.runtime_bindings) | set(
            project.profile.experiment.runtime_context
        )
        self.runtime = {k: v for k, v in available.items() if k in declared and v is not None}
        self.space = SearchSpace(project.profile, runtime_defaults=self.runtime)

    def __enter__(self) -> ProjectSession:
        return self

    def __exit__(self, *args: Any) -> None:
        try:
            if not self.runner_owned_by_tuner:
                self.runner.close()
        finally:
            self.lifecycle.stop()

    def prepare(self, request: EvaluationRequest) -> None:
        if self.runner.PLUGIN_ID == "dry-run":
            return
        changed = self.lifecycle.configure(
            request.engine_id, dict(request.rendered_experiment.get("server_params", {}))
        )
        if not self.started:
            self.execution["host"] = self.lifecycle.start()
            self.started = True
        elif changed or self.project.config.lifecycle.restart_between_evaluations:
            self.execution["host"] = self.lifecycle.restart(
                preserve_data=self.runner.can_reuse_database_state()
            )

    def evaluate(self, candidate: dict[str, Any], index: int) -> dict[str, Any]:
        cfg = self.project.config
        canonical = self.space.canonicalize(candidate)
        request = EvaluationRequest(
            safe_name(f"{cfg.experiment_name}-fixed-{index:05d}"),
            self.project.profile.adapter.engine,
            canonical,
            ExperimentRenderer(self.project.profile).render(canonical, self.runtime),
            self.workload,
            cfg.tuning.seed + index,
            float(cfg.runner.settings.get("timeout_s", 86400)),
        )
        self.prepare(request)
        observation = self.runner.evaluate(request)
        required = {o.metric for o in cfg.tuning.objectives} | {
            c.metric for c in cfg.tuning.all_constraints()
        }
        metrics = dict(observation.metrics)
        valid = observation.ok and all(
            type(metrics.get(k)) in (int, float) and math.isfinite(metrics[k]) for k in required
        )
        if "recall" in metrics and (
            type(metrics["recall"]) not in (int, float) or not 0 <= metrics["recall"] <= 1
        ):
            valid = False
        if "qps" in metrics and (type(metrics["qps"]) not in (int, float) or metrics["qps"] <= 0):
            valid = False
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in metrics.values()
        ):
            valid = False
            metrics = {}  # Invalid JSON numbers must not corrupt persisted artifacts.
        result = {
            "candidate": canonical,
            "status": "ok" if valid else "failed",
            "metrics": metrics,
            "error": observation.error or (None if valid else "missing or invalid metrics"),
            "artifacts": observation.artifacts,
            "auxiliary": observation.auxiliary,
        }
        atomic_write_json(self.project.artifact_dir / "measurements" / f"{index:05d}.json", result)
        return result


def stage_project(project: LoadedProject, directory: Path, stage: str) -> LoadedProject:
    cfg = project.config.model_copy(deep=True)
    cfg.experiment_name = safe_name(f"{cfg.experiment_name}-{stage}", 100)
    cfg.artifact_dir = str(directory)
    return replace(project, config=cfg, artifact_dir=directory)


def tune_project(project: LoadedProject) -> dict[str, Any]:
    started = time.monotonic()
    with ProjectSession(project) as session:
        llm = OpenAICompatibleClient(project.config.llm) if project.config.llm else None
        tuner = Tuner(
            profile=project.profile,
            tuning=project.config.tuning,
            execution=session.execution,
            runner=session.runner,
            artifact_dir=project.artifact_dir,
            experiment_name=project.config.experiment_name,
            llm_client=llm,
            before_evaluation=session.prepare,
            evaluation_timeout_s=float(project.config.runner.settings.get("timeout_s", 86400)),
            proposal_timeout_s=project.config.llm.timeout_s if project.config.llm else 120,
        )
        session.runner_owned_by_tuner = True
        result = tuner.run().to_dict()
    result["invocation_wall_s"] = time.monotonic() - started
    atomic_write_json(project.artifact_dir / "result.json", result)
    return result


def evaluate_candidates(
    project: LoadedProject, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    with ProjectSession(project) as session:
        return [session.evaluate(candidate, i) for i, candidate in enumerate(candidates)]
