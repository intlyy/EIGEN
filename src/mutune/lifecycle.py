"""Safe, explicit service lifecycle controllers used by muTune runners.

The default controller treats the database as an externally managed service.
The Docker Compose controller accepts only a compose file inside a caller-owned
workspace and invokes Docker with argv lists; lifecycle configuration never
contains arbitrary shell commands.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from mutune.errors import ConfigurationError, RunnerError
from mutune.utils import atomic_write_json, ensure_within, fingerprint

_PROJECT_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SERVICE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}\Z")
_POSTGRES_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}\Z")
_FORBIDDEN_COMPOSE_ENV = {
    "COMPOSE_FILE",
    "COMPOSE_PATH_SEPARATOR",
    "COMPOSE_PROJECT_NAME",
    "DOCKER_CONFIG",
    "DOCKER_CONTEXT",
    "DOCKER_HOST",
    "DOCKER_TLS_VERIFY",
    "PATH",
}

_POSTGRES_SETTINGS: dict[str, tuple[str, str]] = {
    "shared_buffers_mb": ("shared_buffers", "size_mb"),
    "effective_cache_size_mb": ("effective_cache_size", "size_mb"),
    "maintenance_work_mem_mb": ("maintenance_work_mem", "size_mb"),
    "max_wal_size_mb": ("max_wal_size", "size_mb"),
    "work_mem_mb": ("work_mem", "size_mb"),
    "effective_io_concurrency": ("effective_io_concurrency", "integer"),
    "random_page_cost": ("random_page_cost", "float"),
    "max_worker_processes": ("max_worker_processes", "integer"),
    "max_parallel_workers": ("max_parallel_workers", "integer"),
    "max_parallel_maintenance_workers": (
        "max_parallel_maintenance_workers",
        "integer",
    ),
}

_MILVUS_SETTINGS: dict[str, tuple[str, ...]] = {
    "root_coord_min_segment_size_to_enable_index": (
        "rootCoord",
        "minSegmentSizeToEnableIndex",
    ),
    "query_coord_auto_handoff": ("queryCoord", "autoHandoff"),
    "query_coord_auto_balance": ("queryCoord", "autoBalance"),
    "common_graceful_time_ms": ("common", "gracefulTime"),
    "data_coord_segment_max_size_mb": ("dataCoord", "segment", "maxSize"),
    "data_coord_segment_seal_proportion": (
        "dataCoord",
        "segment",
        "sealProportion",
    ),
    "data_node_segment_insert_buf_size_bytes": (
        "dataNode",
        "segment",
        "insertBufSize",
    ),
}


@dataclass(frozen=True, slots=True)
class ReadyCheck:
    """Typed readiness policy; no user-supplied command is supported."""

    kind: str
    timeout_s: float = 60.0
    interval_s: float = 0.5
    connect_timeout_s: float = 2.0
    host: str | None = None
    port: int | None = None
    url: str | None = None
    expected_status: tuple[int, ...] = tuple(range(200, 400))

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        endpoint: str,
    ) -> "ReadyCheck":
        if value is None:
            host, port = _endpoint_host_port(endpoint)
            if host is None or port is None:
                return cls(kind="none")
            return cls(kind="tcp", host=host, port=port)

        kind = str(value.get("kind", "tcp")).replace("-", "_")
        if kind not in {"none", "tcp", "http"}:
            raise ConfigurationError(
                f"Unsupported ready-check kind {kind!r}; expected none, tcp, or http"
            )

        timeout_s = _positive_float(value.get("timeout_s", 60.0), "timeout_s")
        interval_s = _positive_float(value.get("interval_s", 0.5), "interval_s")
        connect_timeout_s = _positive_float(
            value.get("connect_timeout_s", 2.0), "connect_timeout_s"
        )

        if kind == "none":
            return cls(
                kind=kind,
                timeout_s=timeout_s,
                interval_s=interval_s,
                connect_timeout_s=connect_timeout_s,
            )

        if kind == "tcp":
            endpoint_host, endpoint_port = _endpoint_host_port(endpoint)
            host = value.get("host", endpoint_host)
            port = value.get("port", endpoint_port)
            if not isinstance(host, str) or not host:
                raise ConfigurationError("TCP ready check requires a non-empty host")
            if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
                raise ConfigurationError("TCP ready check requires port in [1, 65535]")
            return cls(
                kind=kind,
                timeout_s=timeout_s,
                interval_s=interval_s,
                connect_timeout_s=connect_timeout_s,
                host=host,
                port=port,
            )

        url = value.get("url", endpoint)
        if not isinstance(url, str) or urlsplit(url).scheme not in {"http", "https"}:
            raise ConfigurationError("HTTP ready check requires an http:// or https:// URL")
        statuses = value.get("expected_status", tuple(range(200, 400)))
        if isinstance(statuses, int) and not isinstance(statuses, bool):
            statuses = [statuses]
        if not isinstance(statuses, (list, tuple)) or not statuses:
            raise ConfigurationError("expected_status must be a non-empty integer list")
        parsed_statuses: list[int] = []
        for status in statuses:
            if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status <= 599:
                raise ConfigurationError("HTTP expected statuses must be in [100, 599]")
            parsed_statuses.append(status)
        return cls(
            kind=kind,
            timeout_s=timeout_s,
            interval_s=interval_s,
            connect_timeout_s=connect_timeout_s,
            url=url,
            expected_status=tuple(parsed_statuses),
        )

    def wait(self) -> None:
        """Block until ready or raise a bounded RunnerError."""

        if self.kind == "none":
            return

        deadline = time.monotonic() + self.timeout_s
        last_error: BaseException | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = f": {last_error}" if last_error is not None else ""
                raise RunnerError(f"Service did not become ready in {self.timeout_s:g}s{detail}")
            try:
                if self.kind == "tcp":
                    assert self.host is not None and self.port is not None
                    with socket.create_connection(
                        (self.host, self.port),
                        timeout=min(self.connect_timeout_s, remaining),
                    ):
                        return
                assert self.url is not None
                request = urllib.request.Request(self.url, method="GET")
                try:
                    with urllib.request.urlopen(
                        request,
                        timeout=min(self.connect_timeout_s, remaining),
                    ) as response:
                        status = int(response.status)
                except urllib.error.HTTPError as error:
                    status = int(error.code)
                if status in self.expected_status:
                    return
                last_error = RunnerError(f"unexpected HTTP status {status}")
            except (OSError, urllib.error.URLError, RunnerError) as error:
                last_error = error
            time.sleep(min(self.interval_s, max(0.0, deadline - time.monotonic())))


class ServiceLifecycle(ABC):
    """Synchronous lifecycle interface used around one or more evaluations."""

    def __init__(self, endpoint: str, ready_check: ReadyCheck) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ConfigurationError("Service endpoint must be a non-empty string")
        self.endpoint = endpoint.strip()
        self.ready_check = ready_check

    @abstractmethod
    def start(self) -> str:
        """Prepare the service, wait until ready, and return its endpoint."""

    @abstractmethod
    def stop(self) -> None:
        """Release only resources owned by this lifecycle."""

    def configure(
        self,
        engine_id: str,
        server_params: Mapping[str, Any] | None,
    ) -> bool:
        """Apply a typed server configuration and report whether it changed."""

        del engine_id
        if server_params:
            raise ConfigurationError(
                "This lifecycle cannot apply server parameters; use a configured "
                "docker_compose lifecycle"
            )
        return False

    def restart(self, *, preserve_data: bool = False) -> str:
        del preserve_data
        self.stop()
        return self.start()

    def __enter__(self) -> "ServiceLifecycle":
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.stop()


class ExternalLifecycle(ServiceLifecycle):
    """Readiness-only controller for a user-managed database service."""

    def start(self) -> str:
        self.ready_check.wait()
        return self.endpoint

    def stop(self) -> None:
        # External services are never mutated or stopped by muTune.
        return None


class DockerComposeLifecycle(ServiceLifecycle):
    """Own one uniquely named Compose project in an isolated workspace."""

    def __init__(
        self,
        *,
        endpoint: str,
        ready_check: ReadyCheck,
        compose_file: Path,
        workspace_root: Path,
        project_name: str,
        artifact_dir: Path,
        environment: Mapping[str, str] | None = None,
        server_config: Mapping[str, Any] | None = None,
        command_timeout_s: float = 120.0,
        remove_volumes: bool = False,
    ) -> None:
        super().__init__(endpoint, ready_check)
        self.workspace_root = workspace_root.expanduser().resolve()
        self.compose_file = ensure_within(compose_file.expanduser(), self.workspace_root)
        if not self.compose_file.is_file():
            raise ConfigurationError(f"Compose file does not exist: {self.compose_file}")
        if not _PROJECT_NAME.fullmatch(project_name) or not project_name.startswith("mutune-"):
            raise ConfigurationError(
                "Compose project_name must start with 'mutune-' and contain only "
                "lowercase letters, digits, _ and -"
            )
        self.project_name = project_name
        self.artifact_dir = artifact_dir.expanduser().resolve()
        self.command_timeout_s = _positive_float(command_timeout_s, "command_timeout_s")
        self.remove_volumes = bool(remove_volumes)
        self.environment = _validated_environment(environment or {})
        self.server_config = _validated_server_config(server_config)
        self._server_fingerprint: str | None = None
        self._server_state: dict[str, Any] | None = None
        self._override_file: Path | None = None
        self._owned = False

    def _argv(self, *arguments: str) -> list[str]:
        argv = [
            "docker",
            "compose",
            "-f",
            str(self.compose_file),
        ]
        if self._override_file is not None:
            argv.extend(["-f", str(self._override_file)])
        argv.extend(["-p", self.project_name, *arguments])
        return argv

    def configure(
        self,
        engine_id: str,
        server_params: Mapping[str, Any] | None,
    ) -> bool:
        params = dict(server_params or {})
        if not params:
            changed = self._server_fingerprint not in {None, fingerprint({})}
            self._server_fingerprint = fingerprint({})
            self._server_state = None
            self._override_file = None
            return changed
        if self.server_config is None:
            raise ConfigurationError(
                "Rendered candidate contains server_params but lifecycle.settings."
                "server_config is not configured"
            )

        state, override, generated_files = _render_server_configuration(
            engine_id,
            params,
            self.server_config,
            self.artifact_dir,
        )
        new_fingerprint = fingerprint(state, length=64)
        if new_fingerprint == self._server_fingerprint:
            return False

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        for path, contents in generated_files.items():
            path.write_text(contents, encoding="utf-8")
        override_file = self.artifact_dir / "server-compose.override.json"
        atomic_write_json(override_file, override)
        atomic_write_json(
            self.artifact_dir / "server-config.json",
            {
                "engine": engine_id,
                "provider": self.server_config["kind"],
                "parameters": params,
                "fingerprint": new_fingerprint,
                "compose_override": str(override_file),
            },
        )
        self._server_fingerprint = new_fingerprint
        self._server_state = state
        self._override_file = override_file
        return True

    def _run(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(self.environment)
        try:
            completed = subprocess.run(
                self._argv(*arguments),
                cwd=self.compose_file.parent,
                env=environment,
                text=True,
                capture_output=True,
                timeout=self.command_timeout_s,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RunnerError(f"Docker Compose command failed to execute: {error}") from error
        if check and completed.returncode != 0:
            raise RunnerError(
                "Docker Compose command failed "
                f"({completed.returncode}): {' '.join(self._argv(*arguments))}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return completed

    def start(self) -> str:
        if self._owned:
            self.ready_check.wait()
            return self.endpoint
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        completed = self._run("up", "-d", "--remove-orphans")
        self._owned = True
        (self.artifact_dir / "compose-up.json").write_text(
            json.dumps(
                {
                    "argv": self._argv("up", "-d", "--remove-orphans"),
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            self.ready_check.wait()
            self._verify_server_configuration()
        except BaseException:
            self._capture_logs()
            self.stop()
            raise
        return self.endpoint

    def restart(self, *, preserve_data: bool = False) -> str:
        """Recreate the service, optionally retaining its database volume.

        ``docker compose restart`` does not apply a changed command/override.
        ``up --force-recreate`` does, and unlike ``down --volumes`` it keeps
        named and anonymous data volumes attached to the replacement
        container.  The ordinary restart path retains the historical teardown
        behavior for runners that have not explicitly opted into state reuse.
        """

        if not preserve_data:
            return super().restart()
        if not self._owned:
            return self.start()

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        completed = self._run(
            "up",
            "-d",
            "--force-recreate",
            "--remove-orphans",
        )
        atomic_write_json(
            self.artifact_dir / "compose-restart.json",
            {
                "argv": self._argv(
                    "up",
                    "-d",
                    "--force-recreate",
                    "--remove-orphans",
                ),
                "preserved_data": True,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
        try:
            self.ready_check.wait()
            self._verify_server_configuration()
        except BaseException:
            self._capture_logs()
            self.stop()
            raise
        return self.endpoint

    def _verify_server_configuration(self) -> None:
        if self._server_state is None:
            return
        kind = self._server_state["kind"]
        if kind == "postgresql":
            self._verify_postgresql()
            return
        if kind == "milvus_user_yaml":
            self._verify_milvus_user_yaml()
            return
        raise AssertionError(f"unsupported configured server provider: {kind}")

    def _verify_postgresql(self) -> None:
        assert self._server_state is not None
        service = self._server_state["service"]
        user = self._server_state["user"]
        database = self._server_state["database"]
        expected = self._server_state["expected"]
        expressions: list[str] = []
        for key in expected:
            guc_name, kind = _POSTGRES_SETTINGS[key]
            if kind == "size_mb":
                expression = f"pg_size_bytes(current_setting('{guc_name}'))::bigint"
            elif kind == "integer":
                expression = f"current_setting('{guc_name}')::bigint"
            else:
                expression = f"current_setting('{guc_name}')::double precision"
            expressions.extend([f"'{key}'", expression])
        sql = "SELECT json_build_object(" + ",".join(expressions) + ")::text;"
        started_at = time.monotonic()
        deadline = started_at + self.ready_check.timeout_s
        attempts = 0
        last_detail = "PostgreSQL has not reported readiness"
        completed: subprocess.CompletedProcess[str] | None = None
        while True:
            attempts += 1
            ready = self._run(
                "exec",
                "-T",
                service,
                "pg_isready",
                "-U",
                user,
                "-d",
                database,
                "-t",
                str(max(1, math.ceil(self.ready_check.connect_timeout_s))),
                check=False,
            )
            if ready.returncode == 0:
                probe = self._run(
                    "exec",
                    "-T",
                    service,
                    "psql",
                    "-X",
                    "-U",
                    user,
                    "-d",
                    database,
                    "-At",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-c",
                    sql,
                    check=False,
                )
                if probe.returncode == 0:
                    completed = probe
                    break
                last_detail = _compact_process_error("psql", probe)
            else:
                last_detail = _compact_process_error("pg_isready", ready)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RunnerError(
                    "PostgreSQL did not become query-ready inside the container "
                    f"within {self.ready_check.timeout_s:g}s after {attempts} attempts: "
                    f"{last_detail}"
                )
            time.sleep(min(self.ready_check.interval_s, remaining))

        assert completed is not None
        atomic_write_json(
            self.artifact_dir / "postgres-ready.json",
            {
                "attempts": attempts,
                "elapsed_s": time.monotonic() - started_at,
                "service": service,
                "database": database,
            },
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        try:
            actual = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise RunnerError("Could not parse PostgreSQL effective server settings") from error
        mismatches: list[str] = []
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            _native_name, kind = _POSTGRES_SETTINGS[key]
            if kind == "float":
                matches = isinstance(actual_value, (int, float)) and math.isclose(
                    float(actual_value), float(expected_value), rel_tol=1e-9, abs_tol=1e-9
                )
            else:
                matches = actual_value == expected_value
            if not matches:
                mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
        if mismatches:
            raise RunnerError(
                "PostgreSQL server parameters were not applied: " + "; ".join(mismatches)
            )
        atomic_write_json(self.artifact_dir / "effective-server-config.json", actual)

    def _verify_milvus_user_yaml(self) -> None:
        assert self._server_state is not None
        service = self._server_state["service"]
        expected = self._server_state["yaml"]
        completed = self._run(
            "exec",
            "-T",
            service,
            "cat",
            "/milvus/configs/user.yaml",
        )
        if completed.stdout != expected:
            raise RunnerError(
                "Milvus server parameters were not mounted as /milvus/configs/user.yaml"
            )
        (self.artifact_dir / "effective-user.yaml").write_text(
            completed.stdout,
            encoding="utf-8",
        )

    def _capture_logs(self) -> None:
        if not self._owned:
            return
        completed = self._run("logs", "--no-color", check=False)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "compose.log").write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8",
        )

    def stop(self) -> None:
        if not self._owned:
            return
        try:
            self._capture_logs()
        except RunnerError:
            # Diagnostics are best-effort and must never prevent owned service
            # resources from reaching the actual teardown command.
            pass
        arguments = ["down", "--remove-orphans"]
        if self.remove_volumes:
            arguments.append("--volumes")
        try:
            self._run(*arguments)
        finally:
            self._owned = False


def create_lifecycle(
    config: Mapping[str, Any] | None,
    *,
    workspace_root: Path,
    artifact_dir: Path,
    default_endpoint: str,
    default_project_name: str = "mutune-run",
) -> ServiceLifecycle:
    """Construct a lifecycle; absent configuration defaults to external."""

    config = config or {}
    mode = str(config.get("mode", "external")).replace("-", "_")
    settings_value = config.get("settings", {})
    if not isinstance(settings_value, Mapping):
        raise ConfigurationError("lifecycle.settings must be an object")
    settings = dict(settings_value)
    endpoint = settings.get("endpoint", config.get("endpoint", default_endpoint))
    if not isinstance(endpoint, str):
        raise ConfigurationError("lifecycle endpoint must be a string")
    ready_value = settings.get("ready_check", config.get("ready_check"))
    if ready_value is not None and not isinstance(ready_value, Mapping):
        raise ConfigurationError("ready_check must be an object")
    ready_check = ReadyCheck.from_mapping(ready_value, endpoint=endpoint)

    if mode == "external":
        return ExternalLifecycle(endpoint, ready_check)
    if mode != "docker_compose":
        raise ConfigurationError(
            f"Unsupported lifecycle mode {mode!r}; expected external or docker_compose"
        )

    compose_value = settings.get("compose_file")
    if not isinstance(compose_value, str) or not compose_value:
        raise ConfigurationError("docker_compose lifecycle requires settings.compose_file")
    compose_file = Path(compose_value).expanduser()
    if not compose_file.is_absolute():
        compose_file = workspace_root / compose_file
    project_name = settings.get("project_name", default_project_name)
    if not isinstance(project_name, str):
        raise ConfigurationError("project_name must be a string")
    environment = settings.get("environment", {})
    if not isinstance(environment, Mapping):
        raise ConfigurationError("lifecycle environment must be an object")

    return DockerComposeLifecycle(
        endpoint=endpoint,
        ready_check=ready_check,
        compose_file=compose_file,
        workspace_root=workspace_root,
        project_name=project_name,
        artifact_dir=artifact_dir,
        environment={str(key): str(value) for key, value in environment.items()},
        server_config=settings.get("server_config"),
        command_timeout_s=settings.get("command_timeout_s", 120.0),
        remove_volumes=bool(settings.get("remove_volumes", False)),
    )


def _validated_server_config(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConfigurationError("lifecycle.settings.server_config must be an object")
    result = dict(value)
    kind = result.get("kind")
    if kind not in {"postgresql", "milvus_user_yaml"}:
        raise ConfigurationError("server_config.kind must be postgresql or milvus_user_yaml")
    allowed = {"kind", "service"}
    if kind == "postgresql":
        allowed.update({"user", "database"})
    unknown = set(result) - allowed
    if unknown:
        raise ConfigurationError(f"unknown server_config fields for {kind}: {sorted(unknown)}")
    service = result.get("service")
    if not isinstance(service, str) or not _SERVICE_NAME.fullmatch(service):
        raise ConfigurationError("server_config.service is invalid")
    if kind == "postgresql":
        for field_name in ("user", "database"):
            item = result.get(field_name, "postgres")
            if not isinstance(item, str) or not _POSTGRES_IDENTIFIER.fullmatch(item):
                raise ConfigurationError(f"server_config.{field_name} is invalid")
            result[field_name] = item
    return result


def _render_server_configuration(
    engine_id: str,
    server_params: Mapping[str, Any],
    policy: Mapping[str, Any],
    artifact_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[Path, str]]:
    kind = policy["kind"]
    service = policy["service"]
    if kind == "postgresql":
        if engine_id != "pgvector" or set(server_params) != {"postgresql"}:
            raise ConfigurationError(
                "postgresql server_config requires pgvector server_params.postgresql"
            )
        settings = server_params["postgresql"]
        if not isinstance(settings, Mapping):
            raise ConfigurationError("server_params.postgresql must be an object")
        expected, command = _postgresql_command(settings)
        state = {
            "kind": kind,
            "service": service,
            "user": policy["user"],
            "database": policy["database"],
            "expected": expected,
        }
        return state, {"services": {service: {"command": command}}}, {}

    if engine_id != "milvus" or set(server_params) != {"milvus"}:
        raise ConfigurationError(
            "milvus_user_yaml server_config requires milvus server_params.milvus"
        )
    settings = server_params["milvus"]
    if not isinstance(settings, Mapping):
        raise ConfigurationError("server_params.milvus must be an object")
    rendered_yaml = _milvus_user_yaml(settings)
    yaml_path = artifact_dir / "milvus-user.yaml"
    state = {"kind": kind, "service": service, "yaml": rendered_yaml}
    override = {
        "services": {
            service: {
                "volumes": [
                    {
                        "type": "bind",
                        "source": str(yaml_path),
                        "target": "/milvus/configs/user.yaml",
                        "read_only": True,
                    }
                ]
            }
        }
    }
    return state, override, {yaml_path: rendered_yaml}


def _postgresql_command(settings: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    unknown = set(settings) - set(_POSTGRES_SETTINGS)
    if unknown:
        raise ConfigurationError(f"unsupported PostgreSQL server parameters: {sorted(unknown)}")
    if not settings:
        raise ConfigurationError("server_params.postgresql cannot be empty")
    expected: dict[str, Any] = {}
    command = ["postgres"]
    for key in sorted(settings):
        value = settings[key]
        native_name, kind = _POSTGRES_SETTINGS[key]
        if kind in {"size_mb", "integer"}:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError(f"PostgreSQL parameter {key} must be an integer")
            rendered = f"{value}MB" if kind == "size_mb" else str(value)
            expected[key] = value * 1024 * 1024 if kind == "size_mb" else value
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigurationError(f"PostgreSQL parameter {key} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ConfigurationError(f"PostgreSQL parameter {key} must be positive")
            rendered = str(float(value))
            expected[key] = float(value)
        command.extend(["-c", f"{native_name}={rendered}"])
    return expected, command


def _milvus_user_yaml(settings: Mapping[str, Any]) -> str:
    unknown = set(settings) - set(_MILVUS_SETTINGS)
    if unknown:
        raise ConfigurationError(f"unsupported Milvus server parameters: {sorted(unknown)}")
    if set(settings) != set(_MILVUS_SETTINGS):
        missing = set(_MILVUS_SETTINGS) - set(settings)
        raise ConfigurationError(f"missing Milvus server parameters: {sorted(missing)}")

    nested: dict[str, Any] = {}
    for key, path in _MILVUS_SETTINGS.items():
        value = settings[key]
        if key in {"query_coord_auto_handoff", "query_coord_auto_balance"}:
            if type(value) is not bool:
                raise ConfigurationError(f"Milvus parameter {key} must be boolean")
        elif key == "data_coord_segment_seal_proportion":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigurationError(f"Milvus parameter {key} must be numeric")
            if not 0.0 < float(value) < 1.0:
                raise ConfigurationError(f"Milvus parameter {key} must be between 0 and 1")
            value = float(value)
        elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"Milvus parameter {key} must be a positive integer")
        current = nested
        for token in path[:-1]:
            current = current.setdefault(token, {})
        current[path[-1]] = value

    lines = ["# Generated by muTune; candidate-specific Milvus overrides"]
    _append_yaml_lines(lines, nested, indent=0)
    return "\n".join(lines) + "\n"


def _append_yaml_lines(lines: list[str], value: Mapping[str, Any], *, indent: int) -> None:
    prefix = " " * indent
    for key, item in value.items():
        if isinstance(item, Mapping):
            lines.append(f"{prefix}{key}:")
            _append_yaml_lines(lines, item, indent=indent + 2)
        else:
            rendered = str(item).lower() if isinstance(item, bool) else str(item)
            lines.append(f"{prefix}{key}: {rendered}")


def _compact_process_error(
    label: str,
    completed: subprocess.CompletedProcess[str],
) -> str:
    detail = (completed.stderr or completed.stdout).strip()
    detail = " ".join(detail.split())
    if len(detail) > 500:
        detail = detail[:497] + "..."
    suffix = f": {detail}" if detail else ""
    return f"{label} exited with status {completed.returncode}{suffix}"


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{label} must be a positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{label} must be a positive number") from error
    if parsed <= 0:
        raise ConfigurationError(f"{label} must be positive")
    return parsed


def _endpoint_host_port(endpoint: str) -> tuple[str | None, int | None]:
    parsed = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}")
    try:
        port = parsed.port
    except ValueError as error:
        raise ConfigurationError(f"Invalid service endpoint {endpoint!r}: {error}") from error
    if port is None and parsed.scheme in {"http", "https"}:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname, port


def _validated_environment(values: Mapping[str, str]) -> dict[str, str]:
    validated: dict[str, str] = {}
    for key, value in values.items():
        if not _ENVIRONMENT_NAME.fullmatch(key):
            raise ConfigurationError(f"Invalid lifecycle environment name: {key!r}")
        if key in _FORBIDDEN_COMPOSE_ENV or key.startswith("COMPOSE_"):
            raise ConfigurationError(f"Lifecycle may not override sensitive environment {key!r}")
        if "\x00" in value:
            raise ConfigurationError(f"Lifecycle environment {key!r} contains a NUL byte")
        validated[key] = value
    return validated


__all__ = [
    "DockerComposeLifecycle",
    "ExternalLifecycle",
    "ReadyCheck",
    "ServiceLifecycle",
    "create_lifecycle",
]
