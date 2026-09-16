"""Isolated CLI integration for the external vector-db-benchmark project.

The runner never imports vector-db-benchmark modules and never writes to the
configured source repository.  Each evaluation gets a source snapshot beneath
``artifact_dir/workspace`` with its own configuration and result directories.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from mutune.api import BaseRunner, EvaluationRequest, Observation, RunStatus
from mutune.benchmark_compat import (
    MILVUS_GEO_CONTRACT,
    RECALL_CONTRACT,
    install_benchmark_compatibility,
    install_milvus_geo_compatibility,
)
from mutune.errors import ConfigurationError, ResultError, RunnerError
from mutune.utils import atomic_write_json, dataset_sha256, ensure_within, fingerprint, safe_name

_SOURCE_DIRECTORIES = ("benchmark", "engine", "dataset_reader")
_IGNORED_SOURCE_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "results",
    "volumes",
}
_BASE_ENV_NAMES = {"LANG", "LC_ALL", "LC_CTYPE", "PATH", "SYSTEMROOT", "TMPDIR"}
_OVERLAY_PACKAGE = "mutune.resources.vectordb_benchmark"
_PGVECTOR_REBUILD_RESOURCE = "pgvector_rebuild.py.txt"
_MISSING = object()
_ENGINE_OVERLAYS = {
    "pgvector": {
        "pgvector_configure.py.txt": Path("engine/clients/pgvector/configure.py"),
        "pgvector_upload.py.txt": Path("engine/clients/pgvector/upload.py"),
        "pgvector_search.py.txt": Path("engine/clients/pgvector/search.py"),
    }
}


def benchmark_source_sha256(repo: Path) -> str:
    """Fingerprint exactly the upstream code copied into each evaluation."""
    digest = hashlib.sha256()
    paths = [repo / "run.py"]
    for directory in _SOURCE_DIRECTORIES:
        paths.extend(
            p
            for p in (repo / directory).rglob("*")
            if p.is_file()
            and not set(p.relative_to(repo).parts) & _IGNORED_SOURCE_NAMES
            and p.suffix not in {".pyc", ".pyo"}
        )
    for path in sorted(paths):
        _reject_escaping_symlink(path, repo)
        digest.update(path.relative_to(repo).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _link_or_copy_dataset(source: Path, destination: Path) -> None:
    # Windows may not allow unprivileged symlinks. A private copy is safe.
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


class VectorDBBenchmarkRunner(BaseRunner):
    """Run an existing vector-db-benchmark checkout as an external CLI."""

    PLUGIN_ID = "vector-db-benchmark"
    PLUGIN_VERSION = "6"

    def __init__(self, context: Any) -> None:
        super().__init__(context)
        repo_value = context.settings.get("repo_path")
        if not isinstance(repo_value, str) or not repo_value:
            raise ConfigurationError("vector-db-benchmark runner requires settings.repo_path")
        self.repo_root = Path(repo_value).expanduser().resolve()
        _validate_repository(self.repo_root)
        self.source_sha256 = benchmark_source_sha256(self.repo_root)
        expected_source = context.settings.get("expected_source_sha256")
        if expected_source is not None and expected_source != self.source_sha256:
            raise ConfigurationError("benchmark source digest differs from expected_source_sha256")
        self.dataset_digest = None
        dataset_path = context.settings.get("dataset_path")
        self.milvus_geo_filter = context.settings.get("milvus_geo_filter")
        self.max_geo_filter_bytes = context.settings.get("max_geo_filter_bytes", 8 * 1024 * 1024)
        if self.milvus_geo_filter is not None:
            if self.milvus_geo_filter != MILVUS_GEO_CONTRACT:
                raise ConfigurationError("unsupported milvus_geo_filter contract")
            if context.profile.adapter.engine != "milvus" or not context.execution.get("filtered"):
                raise ConfigurationError("milvus_geo_filter requires a filtered Milvus workload")
            if type(self.max_geo_filter_bytes) is not int or self.max_geo_filter_bytes < 1:
                raise ConfigurationError("max_geo_filter_bytes must be a positive integer")
            schema = context.settings.get("dataset_entry", {}).get("schema")
            if not isinstance(schema, dict) or not schema or set(schema.values()) != {"geo"}:
                raise ConfigurationError(
                    "Milvus geo compatibility requires an exclusively geo schema"
                )
            if not dataset_path or not all(
                (Path(dataset_path) / name).is_file()
                for name in ("vectors.npy", "payloads.jsonl", "tests.jsonl")
            ):
                raise ConfigurationError(
                    "Milvus geo requires local vectors, payloads and tests files"
                )
        if dataset_path is not None:
            self.dataset_digest = dataset_sha256(Path(dataset_path))
            expected_data = context.settings.get("expected_dataset_sha256")
            if expected_data is not None and self.dataset_digest != expected_data:
                raise ConfigurationError("dataset content differs from its recorded checksum")

        python_value = context.settings.get("python_executable", sys.executable)
        if not isinstance(python_value, str) or not python_value:
            raise ConfigurationError("python_executable must be a non-empty string")
        self.python_executable = _resolve_executable(python_value)

        self.artifact_root = Path(context.artifact_dir).expanduser().resolve()
        configured_workspace = context.settings.get("workspace_dir")
        if configured_workspace is None:
            self.workspace_root = self.artifact_root / "workspace"
        else:
            if not isinstance(configured_workspace, str) or not configured_workspace:
                raise ConfigurationError("workspace_dir must be a non-empty path string")
            candidate_workspace = Path(configured_workspace).expanduser()
            if not candidate_workspace.is_absolute():
                candidate_workspace = self.artifact_root / candidate_workspace
            try:
                self.workspace_root = ensure_within(candidate_workspace, self.artifact_root)
            except ValueError as error:
                raise ConfigurationError(
                    "workspace_dir must be contained by the artifact directory"
                ) from error

        cache_value = context.settings.get("dataset_cache")
        self.dataset_cache = None
        if cache_value is not None:
            if not isinstance(cache_value, str) or not cache_value:
                raise ConfigurationError("dataset_cache must be a non-empty path string")
            self.dataset_cache = Path(cache_value).expanduser().resolve()

        pass_env = context.settings.get("pass_env", [])
        if not isinstance(pass_env, (list, tuple)) or not all(
            isinstance(name, str) and name for name in pass_env
        ):
            raise ConfigurationError("pass_env must be an environment-variable name list")
        self.pass_env = tuple(pass_env)
        self.keep_workspace = bool(context.settings.get("keep_workspace", True))
        self.max_output_bytes = _positive_int(
            context.settings.get("max_output_bytes", 50 * 1024 * 1024),
            "max_output_bytes",
        )
        self.state_reuse_enabled = bool(context.settings.get("state_reuse", False))
        if any(
            bool(context.settings.get(name, False))
            for name in ("skip_upload", "skip_configure", "skip_search")
        ):
            raise ConfigurationError(
                "Manual skip_upload, skip_configure, and skip_search are unsafe for "
                "configuration evaluation; use verified state_reuse instead"
            )
        self._state_candidate: dict[str, Any] | None = None
        self._state_identity: tuple[Any, ...] | None = None
        self._state_run_id: str | None = None

    def can_reuse_database_state(self) -> bool:
        return self.state_reuse_enabled

    def manifest(self) -> dict[str, Any]:
        return {
            **super().manifest(),
            "state_reuse_enabled": self.state_reuse_enabled,
            "benchmark_source_sha256": self.source_sha256,
            "dataset_sha256": self.dataset_digest,
            "recall_contract": RECALL_CONTRACT,
            "milvus_geo_filter": self.milvus_geo_filter,
            "max_geo_filter_bytes": self.max_geo_filter_bytes if self.milvus_geo_filter else None,
        }

    def close(self) -> None:
        self._invalidate_state()

    def evaluate(self, request: EvaluationRequest) -> Observation:
        if request.timeout_s <= 0 or not math.isfinite(request.timeout_s):
            return Observation(
                status=RunStatus.INVALID,
                error="Evaluation timeout_s must be a positive finite number",
            )

        state_plan = self._state_plan(request)
        print(
            "[muTune] benchmark state: " + _state_plan_message(state_plan),
            file=sys.stderr,
            flush=True,
        )
        if state_plan["mode"] in {"fresh", "rebuild_index"}:
            self._invalidate_state()

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        run_prefix = safe_name(request.run_id, 45)
        workspace = Path(
            tempfile.mkdtemp(prefix=f"{run_prefix}-", dir=self.workspace_root)
        ).resolve()
        raw_root = self.artifact_root / "raw"

        artifacts: list[str] = []
        state_succeeded = False
        evaluation_started = time.monotonic()
        try:
            self._materialize_source(workspace)
            installed_overlays = install_benchmark_compatibility(workspace, request.engine_id)
            installed_overlays += _install_engine_overlays(workspace, request.engine_id)
            if self.milvus_geo_filter:
                if request.engine_id != "milvus" or not request.workload.filtered:
                    raise ConfigurationError("Milvus geo adapter received a different workload")
                installed_overlays += install_milvus_geo_compatibility(workspace)
            self._materialize_dataset(workspace, request.workload.dataset)
            experiment = _prepare_experiment(
                request,
                unique_suffix=fingerprint(workspace.name, length=8),
            )
            if self.milvus_geo_filter:
                experiment["search_params"][0]["mutune_geo"] = {
                    "dataset_path": str(workspace / "datasets/local-data"),
                    "schema": self.context.settings["dataset_entry"]["schema"],
                    "max_filter_bytes": self.max_geo_filter_bytes,
                }
            experiment_name = experiment["name"]
            raw_dir = raw_root / experiment_name
            raw_dir.mkdir(parents=True, exist_ok=False)

            config_dir = workspace / "experiments" / "configurations"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / f"{experiment_name}.json"
            atomic_write_json(config_path, [experiment])
            raw_config_path = raw_dir / "experiment.json"
            atomic_write_json(raw_config_path, [experiment])
            artifacts.append(str(raw_config_path))

            state_path = raw_dir / "state-reuse.json"
            atomic_write_json(state_path, state_plan)
            artifacts.append(str(state_path))

            results_dir = workspace / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = raw_dir / "stdout.log"
            stderr_path = raw_dir / "stderr.log"
            command_path = raw_dir / "command.json"
            preparation_argv: list[str] | None = None
            rebuild_elapsed_s: float | None = None
            if state_plan["mode"] == "rebuild_index":
                helper_path = self._materialize_pgvector_rebuild_helper(workspace)
                rebuild_result_path = raw_dir / "index-rebuild.json"
                rebuild_stdout_path = raw_dir / "index-rebuild.stdout.log"
                rebuild_stderr_path = raw_dir / "index-rebuild.stderr.log"
                preparation_argv = [
                    str(self.python_executable),
                    str(helper_path),
                    "--engine",
                    experiment_name,
                    "--dataset",
                    request.workload.dataset,
                    "--host",
                    self._host(),
                    "--output",
                    str(rebuild_result_path),
                ]
                artifacts.extend(
                    [
                        str(rebuild_result_path),
                        str(rebuild_stdout_path),
                        str(rebuild_stderr_path),
                    ]
                )
                remaining = request.timeout_s - (time.monotonic() - evaluation_started)
                if remaining <= 0:
                    return Observation(
                        status=RunStatus.TIMEOUT,
                        auxiliary={"state_reuse": state_plan},
                        artifacts=artifacts,
                        error="evaluation timed out before index rebuild",
                    )
                returncode, timed_out, _elapsed = _run_process(
                    preparation_argv,
                    cwd=workspace,
                    environment=_sanitized_environment(self.pass_env),
                    stdout_path=rebuild_stdout_path,
                    stderr_path=rebuild_stderr_path,
                    timeout_s=remaining,
                )
                for output_path in (rebuild_stdout_path, rebuild_stderr_path):
                    if _file_size(output_path) > self.max_output_bytes:
                        raise RunnerError("pgvector index-rebuild output exceeded max_output_bytes")
                if timed_out:
                    return Observation(
                        status=RunStatus.TIMEOUT,
                        auxiliary={"state_reuse": state_plan},
                        artifacts=artifacts,
                        error="pgvector index-only rebuild timed out",
                    )
                if returncode != 0:
                    return Observation(
                        status=RunStatus.FAILED,
                        auxiliary={"state_reuse": state_plan},
                        artifacts=artifacts,
                        error=f"pgvector index-only rebuild exited with status {returncode}",
                    )
                rebuild_elapsed_s = _parse_index_rebuild_result(
                    rebuild_result_path,
                    expected_experiment=experiment_name,
                    expected_dataset=request.workload.dataset,
                )
                state_plan["index_rebuild_elapsed_s"] = rebuild_elapsed_s
                atomic_write_json(state_path, state_plan)

            argv = self._command(
                request,
                experiment_name,
                workspace,
                force_skip_upload=state_plan["mode"] != "fresh",
            )
            atomic_write_json(
                command_path,
                {
                    "argv": _redacted_argv(argv),
                    "preparation_argv": (
                        _redacted_argv(preparation_argv) if preparation_argv is not None else None
                    ),
                    "cwd": str(workspace),
                    "timeout_s": request.timeout_s,
                    "source_repo": str(self.repo_root),
                    "compatibility_overlays": installed_overlays,
                },
            )
            artifacts.extend([str(command_path), str(stdout_path), str(stderr_path)])

            remaining = request.timeout_s - (time.monotonic() - evaluation_started)
            if remaining <= 0:
                return Observation(
                    status=RunStatus.TIMEOUT,
                    auxiliary={"state_reuse": state_plan},
                    artifacts=artifacts,
                    error="evaluation timed out before search",
                )
            returncode, timed_out, _search_elapsed_s = _run_process(
                argv,
                cwd=workspace,
                environment=_sanitized_environment(self.pass_env),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout_s=remaining,
            )
            elapsed_s = time.monotonic() - evaluation_started
            if _file_size(stdout_path) > self.max_output_bytes:
                raise RunnerError("vector-db-benchmark stdout exceeded max_output_bytes")
            if _file_size(stderr_path) > self.max_output_bytes:
                raise RunnerError("vector-db-benchmark stderr exceeded max_output_bytes")
            if timed_out:
                return Observation(
                    status=RunStatus.TIMEOUT,
                    auxiliary={"elapsed_s": elapsed_s, "workspace": str(workspace)},
                    artifacts=artifacts,
                    error=f"vector-db-benchmark timed out after {request.timeout_s:g}s",
                )
            if returncode != 0:
                return Observation(
                    status=RunStatus.FAILED,
                    auxiliary={"elapsed_s": elapsed_s, "workspace": str(workspace)},
                    artifacts=artifacts,
                    error=f"vector-db-benchmark exited with status {returncode}",
                )

            search_path = _find_exact_search_result(
                results_dir,
                experiment_name=experiment_name,
                dataset=request.workload.dataset,
            )
            raw_search_path = raw_dir / "search-result.json"
            _copy_regular_file(search_path, raw_search_path, allowed_root=results_dir)
            artifacts.append(str(raw_search_path))

            metrics, auxiliary = parse_vectordb_search_result(
                raw_search_path,
                expected_experiment=experiment_name,
                expected_engine=request.engine_id,
                expected_dataset=request.workload.dataset,
            )
            upload_artifacts, upload_auxiliary = _collect_upload_results(
                results_dir,
                raw_dir,
                experiment_name=experiment_name,
                dataset=request.workload.dataset,
                expected_engine=request.engine_id,
                required=state_plan["mode"] == "fresh",
            )
            artifacts.extend(upload_artifacts)
            auxiliary.update(upload_auxiliary)
            auxiliary["state_reuse"] = dict(state_plan)
            auxiliary["data_reused"] = state_plan["mode"] != "fresh"
            auxiliary["index_reused"] = state_plan["mode"] == "reuse_index"
            if rebuild_elapsed_s is not None:
                auxiliary["build_total_time_s"] = rebuild_elapsed_s
            # These are measured quantities; never manufacture a missing
            # build-time or memory objective from an LLM prediction.
            for name in ("build_total_time_s", "mean_latency_s", "p95_latency_s", "p99_latency_s"):
                if name in auxiliary:
                    metrics[name] = auxiliary[name]
            auxiliary.update(
                {
                    "elapsed_s": elapsed_s,
                    "workspace": str(workspace) if self.keep_workspace else None,
                    "experiment_name": experiment_name,
                    "runner": self.manifest(),
                }
            )
            observation = Observation(
                status=RunStatus.OK,
                metrics=metrics,
                auxiliary=auxiliary,
                artifacts=artifacts,
            )
            self._remember_state(request)
            state_succeeded = True
            return observation
        except (ConfigurationError, ResultError, RunnerError, OSError, ValueError) as error:
            return Observation(
                status=RunStatus.FAILED,
                auxiliary={"workspace": str(workspace)},
                artifacts=artifacts,
                error=str(error),
            )
        finally:
            if not state_succeeded:
                self._invalidate_state()
            if not self.keep_workspace:
                shutil.rmtree(workspace, ignore_errors=True)

    def _state_plan(self, request: EvaluationRequest) -> dict[str, Any]:
        identity = _state_identity(request)
        base = {
            "enabled": self.state_reuse_enabled,
            "mode": "fresh",
            "changed_effects": [],
            "reused_from_run_id": None,
            "reason": "state reuse is disabled",
        }
        if not self.state_reuse_enabled:
            return base
        if request.engine_id != "pgvector":
            base["reason"] = "state reuse is currently implemented only for pgvector"
            return base
        if self._state_candidate is None or self._state_identity != identity:
            base["reason"] = "no compatible successful database state is available"
            return base

        effects = _candidate_effects(
            self.context.profile,
            self._state_candidate,
            request.candidate,
        )
        if effects is None:
            base["reason"] = "candidate effects could not be classified safely"
            return base
        base["changed_effects"] = sorted(effects)
        base["reused_from_run_id"] = self._state_run_id

        if "collection" in effects:
            base["reason"] = "collection changes require a fresh data load"
            return base
        if "index" in effects:
            base["mode"] = "rebuild_index"
            base["reason"] = "uploaded rows are reusable but the vector index changed"
            return base
        known = set(self.context.profile.lifecycle.restart_effects)
        known.update(self.context.profile.lifecycle.reusable_effects)
        if not effects.issubset(known):
            base["reason"] = "profile lifecycle does not declare these effects reusable"
            return base
        base["mode"] = "reuse_index"
        base["reason"] = "uploaded rows and vector index remain compatible"
        return base

    def _remember_state(self, request: EvaluationRequest) -> None:
        if not self.state_reuse_enabled:
            return
        self._state_candidate = dict(request.candidate)
        self._state_identity = _state_identity(request)
        self._state_run_id = request.run_id

    def _invalidate_state(self) -> None:
        self._state_candidate = None
        self._state_identity = None
        self._state_run_id = None

    def _materialize_pgvector_rebuild_helper(self, workspace: Path) -> Path:
        try:
            resource = resources.files(_OVERLAY_PACKAGE).joinpath(_PGVECTOR_REBUILD_RESOURCE)
        except (ModuleNotFoundError, TypeError) as error:
            raise RunnerError(f"Cannot load pgvector rebuild helper: {error}") from error
        if not resource.is_file():
            raise RunnerError("Packaged pgvector index-rebuild helper is missing")
        destination = workspace / "mutune_pgvector_rebuild.py"
        destination.write_bytes(resource.read_bytes())
        return destination

    def _host(self) -> str:
        host = self.context.execution.get(
            "host",
            self.context.settings.get("host", "localhost"),
        )
        if not isinstance(host, str) or not host:
            raise ConfigurationError("execution.host must be a non-empty string")
        return host

    def _materialize_source(self, workspace: Path) -> None:
        for name in _SOURCE_DIRECTORIES:
            source = self.repo_root / name
            _reject_escaping_symlinks(source, self.repo_root)
            shutil.copytree(
                source,
                workspace / name,
                symlinks=False,
                ignore=shutil.ignore_patterns(*_IGNORED_SOURCE_NAMES, "*.pyc", "*.pyo"),
            )
        run_source = self.repo_root / "run.py"
        _reject_escaping_symlink(run_source, self.repo_root)
        shutil.copy2(run_source, workspace / "run.py")

    def _materialize_dataset(self, workspace: Path, dataset_name: str) -> None:
        explicit = self.context.settings.get("dataset_path")
        if explicit is not None:
            source = Path(explicit).expanduser().resolve()
            if not source.exists():
                raise RunnerError(f"Configured dataset_path does not exist: {source}")
            entry = dict(self.context.settings.get("dataset_entry", {}))
            entry.update(
                name=dataset_name,
                vector_size=self.context.execution.get("vector_size"),
                distance=self.context.execution.get("distance"),
                type="tar" if source.is_dir() else "h5",
                path="local-data" if source.is_dir() else "local-data.hdf5",
            )
            if not entry["vector_size"] or entry["distance"] not in {"l2", "cosine", "dot"}:
                raise RunnerError("explicit dataset_path requires vector_size and distance")
            directory = workspace / "datasets"
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write_json(directory / "datasets.json", [entry])
            _link_or_copy_dataset(source, directory / entry["path"])
            return
        source_dataset_dir = self.repo_root / "datasets"
        config_path = source_dataset_dir / "datasets.json"
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RunnerError(f"Failed to read benchmark dataset registry: {error}") from error
        if not isinstance(payload, list):
            raise RunnerError("vector-db-benchmark datasets.json must contain a list")
        matching = [
            entry
            for entry in payload
            if isinstance(entry, Mapping) and entry.get("name") == dataset_name
        ]
        if len(matching) != 1:
            raise RunnerError(
                f"Dataset {dataset_name!r} must have exactly one vector-db-benchmark registry entry"
            )
        entry = json.loads(json.dumps(matching[0]))
        relative_value = entry.get("path")
        if not isinstance(relative_value, str) or not relative_value:
            raise RunnerError(f"Dataset {dataset_name!r} has no valid relative path")
        relative_path = Path(relative_value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RunnerError(f"Dataset {dataset_name!r} uses an unsafe path: {relative_value!r}")

        workspace_dataset_dir = workspace / "datasets"
        workspace_dataset_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(workspace_dataset_dir / "datasets.json", [entry])
        destination = workspace_dataset_dir / relative_path
        source_candidates: list[tuple[Path, Path]] = [
            (source_dataset_dir / relative_path, source_dataset_dir)
        ]
        if self.dataset_cache is not None:
            source_candidates.insert(0, (self.dataset_cache / relative_path, self.dataset_cache))

        for source, allowed_root in source_candidates:
            if not source.exists():
                continue
            try:
                source = ensure_within(source, allowed_root)
            except ValueError as error:
                raise RunnerError(
                    f"Dataset source escapes its configured root: {source}"
                ) from error
            destination.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy_dataset(source, destination)
            return
        # If no staged data exists, the external benchmark may download it into
        # this isolated workspace using the selected registry entry.  It still
        # cannot write into the original repository.

    def _command(
        self,
        request: EvaluationRequest,
        experiment_name: str,
        workspace: Path,
        *,
        force_skip_upload: bool = False,
    ) -> list[str]:
        argv = [
            str(self.python_executable),
            str(workspace / "run.py"),
            "--engines",
            experiment_name,
            "--datasets",
            request.workload.dataset,
            "--host",
            self._host(),
            "--timeout",
            str(request.timeout_s),
        ]
        for setting, flag in (
            ("skip_upload", "--skip-upload"),
            ("skip_search", "--skip-search"),
            ("skip_if_exists", "--skip-if-exists"),
            ("skip_configure", "--skip-configure"),
        ):
            enabled = bool(self.context.settings.get(setting, False))
            if setting == "skip_upload" and force_skip_upload:
                enabled = True
            if enabled:
                argv.append(flag)
        return argv


def _state_identity(request: EvaluationRequest) -> tuple[Any, ...]:
    workload = request.workload
    return (
        request.engine_id,
        workload.dataset,
        workload.distance,
        workload.vector_size,
        workload.top_k,
        workload.concurrency,
        workload.filtered,
        workload.sparse,
    )


def _candidate_effects(
    profile: Any,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> frozenset[str] | None:
    """Classify a canonical candidate diff without revalidating runtime values."""

    try:
        parameters = profile.search_space.parameters
    except AttributeError:
        return None
    changed = {
        name
        for name in set(left) | set(right)
        if left.get(name, _MISSING) != right.get(name, _MISSING)
    }
    if any(name not in parameters for name in changed):
        return None
    return frozenset(parameters[name].effect for name in changed)


def _state_plan_message(plan: Mapping[str, Any]) -> str:
    mode = plan.get("mode")
    if mode == "rebuild_index":
        action = "reusing uploaded rows; rebuilding vector index"
    elif mode == "reuse_index":
        action = "reusing uploaded rows and vector index; running search only"
    else:
        action = "fresh table upload and index build"
    effects = plan.get("changed_effects") or []
    suffix = f"; changed effects={','.join(effects)}" if effects else ""
    return action + suffix


def _parse_index_rebuild_result(
    path: Path,
    *,
    expected_experiment: str,
    expected_dataset: str,
) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultError(f"Failed to parse pgvector index rebuild result: {error}") from error
    if not isinstance(payload, Mapping):
        raise ResultError("pgvector index rebuild result must be an object")
    if payload.get("engine") != "pgvector":
        raise ResultError("pgvector index rebuild result has the wrong engine")
    if payload.get("experiment") != expected_experiment:
        raise ResultError("pgvector index rebuild result has the wrong experiment")
    if payload.get("dataset") != expected_dataset:
        raise ResultError("pgvector index rebuild result has the wrong dataset")
    elapsed = _finite_float(payload.get("elapsed_s"), "index rebuild elapsed_s")
    if elapsed < 0:
        raise ResultError("index rebuild elapsed_s must be non-negative")
    return elapsed


def parse_vectordb_search_result(
    path: Path,
    *,
    expected_experiment: str,
    expected_engine: str,
    expected_dataset: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Strictly parse one vector-db-benchmark search result."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultError(f"Failed to read vector-db-benchmark result {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ResultError("vector-db-benchmark result must be a JSON object")
    params = payload.get("params")
    results = payload.get("results")
    if not isinstance(params, Mapping) or not isinstance(results, Mapping):
        raise ResultError("vector-db-benchmark result requires params and results objects")
    expected = {
        "experiment": expected_experiment,
        "engine": expected_engine,
        "dataset": expected_dataset,
    }
    for key, expected_value in expected.items():
        if params.get(key) != expected_value:
            raise ResultError(
                f"Result {key} mismatch: expected {expected_value!r}, got {params.get(key)!r}"
            )

    qps = _finite_float(results.get("rps"), "results.rps")
    recall = _finite_float(results.get("mean_precisions"), "results.mean_precisions")
    if qps <= 0:
        raise ResultError("results.rps must be positive")
    if not 0.0 <= recall <= 1.0:
        raise ResultError("results.mean_precisions must be in [0, 1]")

    auxiliary: dict[str, Any] = {}
    for source_key, target_key in (
        ("total_time", "total_time_s"),
        ("mean_time", "mean_latency_s"),
        ("p95_time", "p95_latency_s"),
        ("p99_time", "p99_latency_s"),
        ("std_time", "std_latency_s"),
        ("min_time", "min_latency_s"),
        ("max_time", "max_latency_s"),
    ):
        if source_key not in results:
            continue
        parsed = _finite_float(results[source_key], f"results.{source_key}")
        if parsed < 0:
            raise ResultError(f"results.{source_key} must be non-negative")
        auxiliary[target_key] = parsed
    auxiliary["raw_params"] = dict(params)
    return {"qps": qps, "recall": recall}, auxiliary


def _prepare_experiment(
    request: EvaluationRequest,
    *,
    unique_suffix: str,
) -> dict[str, Any]:
    try:
        experiment = json.loads(json.dumps(dict(request.rendered_experiment)))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            f"rendered_experiment must be JSON serializable: {error}"
        ) from error
    if not isinstance(experiment, dict):
        raise ConfigurationError("rendered_experiment must be an object")
    server_params = experiment.pop("server_params", {})
    if not isinstance(server_params, dict):
        raise ConfigurationError("rendered_experiment.server_params must be an object")
    engine = experiment.get("engine")
    if engine != request.engine_id:
        raise ConfigurationError(
            f"rendered engine {engine!r} does not match request engine {request.engine_id!r}"
        )
    for key in ("connection_params", "collection_params", "upload_params"):
        if not isinstance(experiment.get(key), dict):
            raise ConfigurationError(f"rendered_experiment.{key} must be an object")
    search_params = experiment.get("search_params")
    if not isinstance(search_params, list) or len(search_params) != 1:
        raise ConfigurationError("muTune requires exactly one search_params entry per evaluation")
    if not isinstance(search_params[0], dict):
        raise ConfigurationError("rendered_experiment.search_params[0] must be an object")
    search_params[0].setdefault("parallel", request.workload.concurrency)
    search_params[0].setdefault("top", request.workload.top_k)

    digest = fingerprint(
        {
            "engine": request.engine_id,
            "candidate": dict(request.candidate),
            "rendered": experiment,
            "seed": request.seed,
        },
        length=12,
    )
    experiment["name"] = (
        f"mutune-{safe_name(request.run_id, 55)}-{digest}-{safe_name(unique_suffix, 12)}"
    )
    return experiment


def _install_engine_overlays(workspace: Path, engine_id: str) -> list[str]:
    """Install packaged adapter compatibility files into an isolated snapshot.

    The configured source checkout is never changed.  Overlay destinations are
    fixed package-owned paths, and every destination must already exist in the
    copied benchmark source so an incompatible checkout fails closed.
    """

    overlays = _ENGINE_OVERLAYS.get(engine_id)
    if overlays is None:
        return []

    try:
        resource_root = resources.files(_OVERLAY_PACKAGE)
    except (ModuleNotFoundError, TypeError) as error:
        raise RunnerError(
            f"Cannot load packaged compatibility overlays for {engine_id!r}: {error}"
        ) from error

    installed: list[str] = []
    for resource_name, relative_destination in overlays.items():
        resource = resource_root.joinpath(resource_name)
        if not resource.is_file():
            raise RunnerError(f"Packaged compatibility overlay is missing: {resource_name}")
        try:
            destination = ensure_within(workspace / relative_destination, workspace)
        except ValueError as error:  # pragma: no cover - mappings are package constants
            raise RunnerError(
                f"Compatibility overlay destination escapes workspace: {relative_destination}"
            ) from error
        if destination.is_symlink() or not destination.is_file():
            raise RunnerError(
                "The vector-db-benchmark checkout is incompatible with the "
                f"{engine_id!r} overlay; expected a regular file at "
                f"{relative_destination.as_posix()}"
            )
        try:
            destination.write_bytes(resource.read_bytes())
        except OSError as error:
            raise RunnerError(
                f"Failed to install compatibility overlay {resource_name}: {error}"
            ) from error
        installed.append(relative_destination.as_posix())
    return installed


def _validate_repository(repo_root: Path) -> None:
    if not repo_root.is_dir():
        raise ConfigurationError(f"vector-db-benchmark repo does not exist: {repo_root}")
    required = [repo_root / "run.py", repo_root / "datasets" / "datasets.json"]
    required.extend(repo_root / name for name in _SOURCE_DIRECTORIES)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ConfigurationError(
            "Invalid vector-db-benchmark repository; missing: " + ", ".join(missing)
        )
    for path in required:
        _reject_escaping_symlink(path, repo_root)


def _resolve_executable(value: str) -> Path:
    if os.path.isabs(value) or os.sep in value:
        executable = Path(value).expanduser().resolve()
        if not executable.is_file():
            raise ConfigurationError(f"Python executable does not exist: {executable}")
        return executable
    located = shutil.which(value)
    if located is None:
        raise ConfigurationError(f"Python executable is not on PATH: {value}")
    return Path(located).resolve()


def _run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
) -> tuple[int, bool, float]:
    started = time.monotonic()
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                shell=False,
                start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_group(process)
                process.wait()
            except BaseException:
                _terminate_process_group(process)
                process.wait()
                raise
    except OSError as error:
        raise RunnerError(f"Failed to execute vector-db-benchmark: {error}") from error
    return process.returncode, timed_out, time.monotonic() - started


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows CI is not used by this project
            process.kill()
    except ProcessLookupError:
        return


def _find_exact_search_result(
    results_dir: Path,
    *,
    experiment_name: str,
    dataset: str,
) -> Path:
    prefix = f"{experiment_name}-{dataset}-search-0-"
    candidates = [
        path
        for path in results_dir.iterdir()
        if path.name.startswith(prefix) and path.name.endswith(".json") and path.is_file()
    ]
    if len(candidates) != 1:
        raise ResultError(
            f"Expected exactly one search result for {experiment_name!r}, found {len(candidates)}"
        )
    if candidates[0].is_symlink():
        raise ResultError("Search result may not be a symlink")
    return candidates[0]


def _collect_upload_results(
    results_dir: Path,
    raw_dir: Path,
    *,
    experiment_name: str,
    dataset: str,
    expected_engine: str,
    required: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    prefix = f"{experiment_name}-{dataset}-upload-"
    candidates = [
        path
        for path in results_dir.iterdir()
        if path.name.startswith(prefix) and path.name.endswith(".json") and path.is_file()
    ]
    if len(candidates) > 1:
        raise ResultError(
            f"Expected at most one upload result for {experiment_name!r}, found {len(candidates)}"
        )
    if not candidates:
        if required:
            raise ResultError("Fresh evaluation requires a matching upload/build result")
        return [], {}
    if candidates[0].is_symlink():
        raise ResultError("Upload result may not be a symlink")
    destination = raw_dir / "upload-result.json"
    _copy_regular_file(candidates[0], destination, allowed_root=results_dir)
    try:
        payload = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResultError(f"Failed to parse upload result: {error}") from error
    if not isinstance(payload, Mapping):
        raise ResultError("Upload result must be an object")
    params = payload.get("params")
    results = payload.get("results")
    if not isinstance(params, Mapping) or not isinstance(results, Mapping):
        raise ResultError("Upload result requires params and results objects")
    expected = {
        "experiment": experiment_name,
        "engine": expected_engine,
        "dataset": dataset,
    }
    for key, expected_value in expected.items():
        if params.get(key) != expected_value:
            raise ResultError(f"Upload result {key} does not match the request")
    auxiliary: dict[str, Any] = {}
    if required and "total_time" not in results:
        raise ResultError("Fresh upload/build result requires results.total_time")
    for key, target in (("upload_time", "upload_time_s"), ("total_time", "build_total_time_s")):
        if key in results:
            parsed = _finite_float(results[key], f"upload results.{key}")
            if parsed < 0:
                raise ResultError(f"upload results.{key} must be non-negative")
            auxiliary[target] = parsed
    return [str(destination)], auxiliary


def _copy_regular_file(source: Path, destination: Path, *, allowed_root: Path) -> None:
    try:
        resolved = ensure_within(source, allowed_root)
    except ValueError as error:
        raise ResultError(f"Result path escapes its results directory: {source}") from error
    if source.is_symlink() or not resolved.is_file():
        raise ResultError(f"Result is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved, destination)


def _reject_escaping_symlink(path: Path, allowed_root: Path) -> None:
    if not path.is_symlink():
        return
    try:
        ensure_within(path, allowed_root)
    except ValueError as error:
        raise ConfigurationError(f"Source symlink escapes repository: {path}") from error


def _reject_escaping_symlinks(root: Path, allowed_root: Path) -> None:
    _reject_escaping_symlink(root, allowed_root)
    for path in root.rglob("*"):
        _reject_escaping_symlink(path, allowed_root)


def _sanitized_environment(pass_env: Sequence[str]) -> dict[str, str]:
    allowed = set(_BASE_ENV_NAMES)
    allowed.update(pass_env)
    return {name: os.environ[name] for name in allowed if name in os.environ}


def _redacted_argv(argv: Sequence[str]) -> list[str]:
    redacted = list(argv)
    try:
        host_index = redacted.index("--host") + 1
    except ValueError:
        return redacted
    if host_index < len(redacted):
        redacted[host_index] = _redact_endpoint(redacted[host_index])
    return redacted


def _redact_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.username is None and parsed.password is None:
        return endpoint
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit(
        (parsed.scheme, f"***:***@{host}", parsed.path, parsed.query, parsed.fragment)
    )


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ResultError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ResultError(f"{label} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ResultError(f"{label} must be finite")
    return parsed


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


__all__ = [
    "VectorDBBenchmarkRunner",
    "parse_vectordb_search_result",
]
