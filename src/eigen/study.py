"""Paper outer loop: parallel MiniDB tuning, cross-validation, full-DB transfer."""

from __future__ import annotations

import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from eigen.benchmark_compat import MILVUS_GEO_CONTRACT
from eigen.config import LoadedProject, StrictModel, load_project
from eigen.execution import evaluate_candidates, stage_project, tune_project
from eigen.profiles import profile_fingerprint
from eigen.timing import finalize_study_timings
from eigen.tuning.history import candidate_key
from eigen.tuning.pareto import feasible
from eigen.utils import atomic_write_json, fingerprint
from eigen.worker import RemoteWorker, run_remote_worker


class StudyConfig(StrictModel):
    schema_version: Literal[1] = 1
    name: str = "eigen-study"
    minidbs: list[str] = Field(min_length=1)
    full_database: str
    minidb_manifest: str
    artifact_dir: str = "./artifacts/study"
    top_l: int = Field(default=5, gt=0)
    stability_weight: float = Field(default=1.0, ge=0)
    remote_workers: list[RemoteWorker] | None = None

    @model_validator(mode="after")
    def distinct_projects(self) -> StudyConfig:
        if len(set(self.minidbs)) != len(self.minidbs):
            raise ValueError("minidbs must reference distinct project configs")
        if self.remote_workers is not None:
            if len(self.remote_workers) != len(self.minidbs):
                raise ValueError("remote_workers must contain one entry per MiniDB in order")
            hosts = [worker.host.rsplit("@", 1)[-1].lower() for worker in self.remote_workers]
            if len(set(hosts)) != len(hosts):
                raise ValueError("remote MiniDB workers require separate machines/SSH host aliases")
        return self


@dataclass(frozen=True)
class LoadedStudy:
    config: StudyConfig
    minidbs: list[LoadedProject]
    full_database: LoadedProject
    artifact_dir: Path
    manifest: dict[str, Any]


def load_study(path: str | Path) -> LoadedStudy:
    source = Path(path).expanduser().resolve()
    cfg = StudyConfig.model_validate_json(source.read_text(encoding="utf-8"))
    projects = [load_project(source.parent / ref) for ref in cfg.minidbs]
    full = load_project(source.parent / cfg.full_database)
    manifest_path = (source.parent / cfg.minidb_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_study_projects(projects, full, manifest, manifest_path.parent, cfg.remote_workers)
    return LoadedStudy(cfg, projects, full, (source.parent / cfg.artifact_dir).resolve(), manifest)


def validate_study_projects(
    projects: list[LoadedProject],
    full: LoadedProject,
    manifest: dict[str, Any],
    manifest_dir: Path,
    remote_workers: list[RemoteWorker] | None = None,
) -> None:
    """Check identities before starting any expensive or stateful work."""
    all_projects = [*projects, full]
    if len({profile_fingerprint(p.profile) for p in all_projects}) != 1:
        raise ValueError("all MiniDBs and the full database must share one exact engine profile")
    workload_keys = ("distance", "vector_size", "top_k", "search_parallel", "filtered", "sparse")
    for key in workload_keys:
        if len({getattr(p.config.execution, key) for p in all_projects}) != 1:
            raise ValueError(f"cross-view workload mismatch: {key}")
    contracts = [
        fingerprint(
            {
                "objectives": [o.model_dump() for o in p.config.tuning.objectives],
                "constraints": [c.model_dump() for c in p.config.tuning.all_constraints()],
                "primary": p.config.tuning.objective_metric,
            }
        )
        for p in all_projects
    ]
    if len(set(contracts)) != 1:
        raise ValueError("cross-view objective/constraint mismatch")
    # The benchmark uses fixed table/collection names, so separate ports/hosts
    # and owned Compose project names are mandatory for concurrent workers.
    endpoints = []
    for i, project in enumerate(projects):
        execution = project.config.execution
        host = project.config.lifecycle.settings.get("endpoint", execution.host)
        address = urlsplit(host if "://" in host else "//" + host)
        hostname = address.hostname
        if hostname in {"localhost", "127.0.0.1", "::1"}:
            hostname = "loopback"
        defaults = {"milvus": 19530, "qdrant": 6333, "pgvector": 5432}
        port = execution.connection_params.get(
            "port", address.port or defaults.get(project.profile.adapter.engine)
        )
        machine = remote_workers[i].host if remote_workers else "local"
        endpoints.append((machine, hostname, port))
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("parallel MiniDB workers require distinct database endpoints")
    owned = [
        (
            remote_workers[i].host if remote_workers else "local",
            p.config.lifecycle.settings.get("project_name", p.config.experiment_name),
        )
        for i, p in enumerate(projects)
        if p.config.lifecycle.mode == "docker_compose"
    ]
    if len(set(owned)) != len(owned):
        raise ValueError("parallel Compose workers require distinct project names")
    if len({p.config.tuning.seed for p in projects}) != len(projects):
        raise ValueError("MiniDB tuning seeds must be independent and distinct")
    entries = manifest.get("minidbs", [])
    if len(entries) != len(projects) or not manifest.get("bucket_fingerprint"):
        raise ValueError(
            "MiniDB manifest must describe all views and their common bucket fingerprint"
        )
    if len({entry["sample_seed"] for entry in entries}) != len(entries):
        raise ValueError("MiniDB sampling seeds must be distinct")
    for project, entry in zip(projects, entries, strict=True):
        dataset = project.config.runner.settings.get("dataset_path")
        expected = Path(entry["path"])
        if not expected.is_absolute():
            expected = manifest_dir / expected
        if dataset is None or Path(dataset).resolve() != expected.resolve():
            raise ValueError("MiniDB project dataset_path must match its manifest entry in order")
        if not expected.exists() or not entry.get("sha256"):
            raise ValueError("MiniDB data and its manifest checksum must exist")
        project.config.runner.settings["expected_dataset_sha256"] = entry["sha256"]
    original = full.config.runner.settings.get("dataset_path")
    if original is None or Path(original).resolve() != Path(manifest["source"]).resolve():
        raise ValueError(
            "full_database must use the source dataset recorded in the MiniDB manifest"
        )
    if not Path(original).exists() or not manifest.get("source_sha256"):
        raise ValueError("source dataset and its manifest checksum must exist")
    full.config.runner.settings["expected_dataset_sha256"] = manifest["source_sha256"]
    if (
        manifest.get("dimension") != full.config.execution.vector_size
        or manifest.get("metric") != full.config.execution.distance
    ):
        raise ValueError("MiniDB manifest dimension/distance differs from configured workload")
    if manifest.get("top_k", 0) < full.config.execution.top_k:
        raise ValueError("MiniDB ground truth has fewer neighbors than workload top_k")
    if manifest.get("format") == "geo":
        engine = full.profile.adapter.engine
        if not full.config.execution.filtered or engine not in {"milvus", "qdrant"}:
            raise ValueError("geo manifest requires a filtered Milvus or Qdrant workload")
        if engine == "milvus":
            for project in all_projects:
                settings = project.config.runner.settings
                if settings.get("milvus_geo_filter") != MILVUS_GEO_CONTRACT:
                    raise ValueError(
                        "Milvus geo requires the explicit payload-ID prefilter contract"
                    )
                schema = settings.get("dataset_entry", {}).get("schema")
                if schema != manifest.get("payload_schema") or not schema:
                    raise ValueError("Milvus geo payload schema must match the MiniDB manifest")
    for project in all_projects:
        if project.config.runner.settings.get("state_reuse") and any(
            o.metric == "build_total_time_s" for o in project.config.tuning.objectives
        ):
            raise ValueError("build-time optimization requires fresh builds: disable state_reuse")


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    # A constant dimension gives every candidate equal contribution.
    return [(v - lo) / (hi - lo) for v in values] if hi > lo else [0.0] * len(values)


def rank_cross_validation(
    candidates: list[dict[str, Any]],
    matrix: list[list[dict[str, Any]]],
    *,
    constraints: Any,
    objective_metric: str = "qps",
    stability_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Matrix rows are MiniDBs, columns are deduplicated candidates (Eq. 2–5)."""
    if not matrix or any(len(row) != len(candidates) for row in matrix):
        raise ValueError("cross-validation requires every candidate on every MiniDB")
    keep = [
        i
        for i in range(len(candidates))
        if all(
            row[i]["status"] == "ok"
            and feasible(row[i]["metrics"], constraints)
            and type(row[i]["metrics"].get(objective_metric)) in (int, float)
            and math.isfinite(row[i]["metrics"][objective_metric])
            for row in matrix
        )
    ]
    normalized = [minmax([row[i]["metrics"][objective_metric] for i in keep]) for row in matrix]
    means = [statistics.fmean(row[j] for row in normalized) for j in range(len(keep))]
    stds = [statistics.pstdev(row[j] for row in normalized) for j in range(len(keep))]
    performance, variability = minmax(means), minmax(stds)
    result = [
        {
            "candidate": candidates[i],
            "candidate_key": candidate_key(candidates[i]),
            "mean": means[j],
            "stddev": stds[j],
            "performance_score": performance[j],
            "stability_score": 1 - variability[j],
            "score": performance[j] + stability_weight * (1 - variability[j]),
            "normalized_qps": [row[j] for row in normalized],
            "measurements": [row[i] for row in matrix],
        }
        for j, i in enumerate(keep)
    ]
    return sorted(result, key=lambda item: (-item["score"], -item["mean"], item["candidate_key"]))


def _run_study(study: LoadedStudy, *, tune_fn: Any, evaluate_fn: Any) -> dict[str, Any]:
    cfg, root = study.config, study.artifact_dir
    root.mkdir(parents=True, exist_ok=True)
    # A stage run never silently consumes an old cross-validation matrix.
    # Local Tuner histories independently resume under their strict contracts.
    contract = {
        "study": cfg.model_dump(
            exclude={"remote_workers"} if cfg.remote_workers is None else set()
        ),
        "minidb_manifest": study.manifest,
        "projects": [
            fingerprint(p.config.model_dump(mode="json"), length=64)
            for p in [*study.minidbs, study.full_database]
        ],
        "profile": profile_fingerprint(study.full_database.profile),
    }
    contract_path = root / "study_manifest.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise ValueError("study artifacts belong to a different config; use a new artifact_dir")
    atomic_write_json(contract_path, contract)
    atomic_write_json(root / "result.json", {"status": "running", "best_candidate": None})
    timings = {
        "cross_validation_worker_wall_s": [0.0] * len(study.minidbs),
        "cross_validation_wall_s": 0.0,
        "aggregation_wall_s": 0.0,
        "full_validation_wall_s": 0.0,
    }
    local_projects = [
        stage_project(p, root / f"mini-{i:02d}" / "tuning", "tuning")
        for i, p in enumerate(study.minidbs)
    ]
    start = time.monotonic()

    remote_run_id = fingerprint(contract)

    def timed_tune(indexed_project):
        i, project = indexed_project
        worker_started = time.monotonic()
        if cfg.remote_workers is None:
            local_result = tune_fn(project)
            worker_wall_s = time.monotonic() - worker_started
        else:
            local_result, worker_wall_s = run_remote_worker(
                cfg.remote_workers[i],
                study.minidbs[i],
                project.artifact_dir,
                stage="tuning",
                run_id=remote_run_id,
            )
        finished = time.monotonic()
        return local_result, finished, worker_wall_s

    with ThreadPoolExecutor(max_workers=len(local_projects)) as pool:
        completed = list(pool.map(timed_tune, enumerate(local_projects)))
    timings["parallel_tuning_wall_s"] = time.monotonic() - start
    timings["critical_path_worker"] = max(range(len(completed)), key=lambda i: completed[i][1])
    timings["local_optimization_worker_wall_s"] = [item[2] for item in completed]
    local_results = [item[0] for item in completed]
    if any(not result.get("complete") for result in local_results):
        raise ValueError("all independent MiniDB tuning budgets must finish before transfer")
    aggregation_started = time.monotonic()
    union = {
        candidate_key(item["candidate"]): item["candidate"]
        for result in local_results
        for item in result["transfer_candidates"]
    }
    candidates = [union[key] for key in sorted(union)]
    timings["aggregation_wall_s"] += time.monotonic() - aggregation_started
    atomic_write_json(root / "candidate_pool.json", candidates)
    result: dict[str, Any] = {
        "status": "no_feasible_candidates",
        "metrics_origin": (
            "synthetic"
            if any(
                p.config.runner.plugin == "dry-run" for p in [*study.minidbs, study.full_database]
            )
            else "measured"
        ),
        "best_candidate": None,
        "best_metrics": None,
        "local_results": local_results,
        "ranking": [],
        "full_evaluations": [],
        "timings": timings,
        "cross_validation_evaluations": 0,
        "full_validation_evaluations": 0,
    }
    if candidates:
        validation_projects = [
            stage_project(p, root / f"mini-{i:02d}" / "validation", "validation")
            for i, p in enumerate(study.minidbs)
        ]
        start = time.monotonic()

        def timed_cross_validation(indexed_project):
            i, project = indexed_project
            worker_started = time.monotonic()
            if cfg.remote_workers is not None:
                return run_remote_worker(
                    cfg.remote_workers[i],
                    study.minidbs[i],
                    project.artifact_dir,
                    stage="validation",
                    run_id=remote_run_id,
                    candidates=candidates,
                )
            measurements = evaluate_fn(project, candidates)
            return measurements, time.monotonic() - worker_started

        with ThreadPoolExecutor(max_workers=len(validation_projects)) as pool:
            cross_completed = list(pool.map(timed_cross_validation, enumerate(validation_projects)))
        timings["cross_validation_wall_s"] = time.monotonic() - start
        timings["cross_validation_worker_wall_s"] = [item[1] for item in cross_completed]
        matrix = [item[0] for item in cross_completed]
        result["cross_validation_evaluations"] = sum(len(row) for row in matrix)
        atomic_write_json(
            root / "cross_validation.json", {"candidates": candidates, "matrix": matrix}
        )
        tuning = study.full_database.config.tuning
        aggregation_started = time.monotonic()
        ranking = rank_cross_validation(
            candidates,
            matrix,
            constraints=tuning.all_constraints(),
            objective_metric=tuning.objective_metric,
            stability_weight=cfg.stability_weight,
        )
        timings["aggregation_wall_s"] += time.monotonic() - aggregation_started
        result["ranking"] = ranking
        atomic_write_json(root / "ranking.json", ranking)
        if ranking:
            full = stage_project(study.full_database, root / "full", "transfer")
            start = time.monotonic()
            measurements = evaluate_fn(full, [item["candidate"] for item in ranking[: cfg.top_l]])
            timings["full_validation_wall_s"] = time.monotonic() - start
            result["full_evaluations"] = measurements
            result["full_validation_evaluations"] = len(measurements)
            valid = [
                item
                for item in measurements
                if item["status"] == "ok"
                and feasible(item["metrics"], tuning.all_constraints())
                and type(item["metrics"].get(tuning.objective_metric)) in (int, float)
                and math.isfinite(item["metrics"][tuning.objective_metric])
            ]
            if valid:
                best = max(valid, key=lambda item: item["metrics"][tuning.objective_metric])
                result.update(
                    status="ok", best_candidate=best["candidate"], best_metrics=best["metrics"]
                )
            else:
                result["status"] = "no_full_database_feasible_candidate"
    atomic_write_json(root / "result.json", result)
    return result


def run_study(
    study: LoadedStudy, *, tune_fn: Any = tune_project, evaluate_fn: Any = evaluate_candidates
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = _run_study(study, tune_fn=tune_fn, evaluate_fn=evaluate_fn)
        finalize_study_timings(
            result["timings"], study.manifest, result["local_results"], time.monotonic() - started
        )
        atomic_write_json(study.artifact_dir / "result.json", result)
    except BaseException as error:
        # Retain measurement artifacts, but do not leave a previous successful
        # recommendation looking like the outcome of an interrupted invocation.
        atomic_write_json(
            study.artifact_dir / "invocation_error.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "wall_s": time.monotonic() - started,
            },
        )
        atomic_write_json(
            study.artifact_dir / "result.json",
            {
                "status": "failed",
                "best_candidate": None,
                "best_metrics": None,
                "error_type": type(error).__name__,
            },
        )
        raise
    return result
