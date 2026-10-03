"""One-shot, JSON-over-stdio runner protocol for out-of-process plugins.

The transport isolates dependency environments, but it is not a security
sandbox.  Only trusted installed plugins should expose this runner; untrusted
executables additionally need an OS/container sandbox outside this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from eigen.api import BaseRunner, EvaluationRequest, Observation, RunStatus
from eigen.errors import ConfigurationError, ResultError
from eigen.utils import atomic_write_json, ensure_within, fingerprint, safe_name

PROTOCOL_NAME = "eigen-evaluation"
PROTOCOL_VERSION = "1.0"


class OneShotSubprocessRunner(BaseRunner):
    """Execute one plugin process per evaluation.

    ``RunnerContext.settings.command`` must be an argv list.  It is never
    interpreted by a shell.  This class is intentionally not a built-in
    registry entry: a trusted package may register a configured subclass in
    ``eigen.runners`` or construct it explicitly.
    """

    PLUGIN_ID = "subprocess-one-shot"
    PLUGIN_VERSION = "1"

    def __init__(self, context: Any) -> None:
        super().__init__(context)
        command = context.settings.get("command")
        if not isinstance(command, (list, tuple)) or not command:
            raise ConfigurationError("subprocess runner requires settings.command as an argv list")
        self.command = _validate_command(command, context.settings.get("executable_roots"))
        self.append_evaluate_arg = bool(context.settings.get("append_evaluate_arg", True))
        self.max_stdout_bytes = _positive_int(
            context.settings.get("max_stdout_bytes", 1_048_576), "max_stdout_bytes"
        )
        self.max_stderr_bytes = _positive_int(
            context.settings.get("max_stderr_bytes", 10_485_760), "max_stderr_bytes"
        )
        required = context.settings.get("required_metrics", ["qps", "recall"])
        if not isinstance(required, (list, tuple)) or not all(
            isinstance(item, str) and item for item in required
        ):
            raise ConfigurationError("required_metrics must be a string list")
        self.required_metrics = tuple(required)
        pass_env = context.settings.get("pass_env", [])
        if not isinstance(pass_env, (list, tuple)) or not all(
            isinstance(item, str) and item for item in pass_env
        ):
            raise ConfigurationError("pass_env must be an environment-variable name list")
        self.pass_env = tuple(pass_env)

    def evaluate(self, request: EvaluationRequest) -> Observation:
        root = self.context.artifact_dir.expanduser().resolve() / "subprocess"
        root.mkdir(parents=True, exist_ok=True)
        run_prefix = safe_name(request.run_id, 50)
        run_dir = Path(tempfile.mkdtemp(prefix=f"{run_prefix}-", dir=root)).resolve()
        protocol_request = _request_payload(request, run_dir)
        request_path = run_dir / "request.json"
        stdout_path = run_dir / "response.json"
        stderr_path = run_dir / "stderr.log"
        atomic_write_json(request_path, protocol_request)

        argv = list(self.command)
        if self.append_evaluate_arg and (not argv or argv[-1] != "evaluate"):
            argv.append("evaluate")
        environment = _sanitized_environment(self.pass_env)
        payload = json.dumps(protocol_request, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )

        timed_out = False
        returncode: int | None = None
        try:
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    cwd=run_dir,
                    env=environment,
                    shell=False,
                    start_new_session=True,
                )
                try:
                    process.communicate(input=payload, timeout=request.timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_group(process)
                    process.wait()
                returncode = process.returncode
        except OSError as error:
            return Observation(
                status=RunStatus.FAILED,
                artifacts=[str(request_path), str(stdout_path), str(stderr_path)],
                error=f"Failed to execute subprocess plugin: {error}",
            )

        artifacts = [str(request_path), str(stdout_path), str(stderr_path)]
        if timed_out:
            return Observation(
                status=RunStatus.TIMEOUT,
                artifacts=artifacts,
                error=f"Subprocess plugin timed out after {request.timeout_s:g}s",
            )
        if returncode != 0:
            return Observation(
                status=RunStatus.FAILED,
                artifacts=artifacts,
                error=f"Subprocess plugin exited with status {returncode}",
            )
        if stdout_path.stat().st_size > self.max_stdout_bytes:
            return Observation(
                status=RunStatus.FAILED,
                artifacts=artifacts,
                error="Subprocess plugin response exceeded max_stdout_bytes",
            )
        if stderr_path.stat().st_size > self.max_stderr_bytes:
            return Observation(
                status=RunStatus.FAILED,
                artifacts=artifacts,
                error="Subprocess plugin stderr exceeded max_stderr_bytes",
            )

        try:
            response = json.loads(stdout_path.read_text(encoding="utf-8"))
            observation, plugin_artifacts = _parse_response(
                response,
                request=request,
                run_dir=run_dir,
                required_metrics=self.required_metrics,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ResultError) as error:
            return Observation(
                status=RunStatus.FAILED,
                artifacts=artifacts,
                error=f"Invalid subprocess plugin response: {error}",
            )
        observation.artifacts = artifacts + plugin_artifacts
        return observation


# Public alias with the terminology used in configuration and documentation.
SubprocessPluginRunner = OneShotSubprocessRunner


def _request_payload(request: EvaluationRequest, run_dir: Path) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request.run_id,
        "request_fingerprint": fingerprint(
            {
                "engine": request.engine_id,
                "candidate": dict(request.candidate),
                "workload": asdict(request.workload),
                "seed": request.seed,
            }
        ),
        "run_dir": str(run_dir),
        "engine": {"id": request.engine_id},
        "workload": asdict(request.workload),
        "candidate": dict(request.candidate),
        "rendered_experiment": dict(request.rendered_experiment),
        "seed": request.seed,
        "timeout_s": request.timeout_s,
    }


def _parse_response(
    value: Any,
    *,
    request: EvaluationRequest,
    run_dir: Path,
    required_metrics: Sequence[str],
) -> tuple[Observation, list[str]]:
    if not isinstance(value, Mapping):
        raise ResultError("response must be a JSON object")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ResultError(
            f"protocol_version must be {PROTOCOL_VERSION!r}, got {value.get('protocol_version')!r}"
        )
    if value.get("request_id") != request.run_id:
        raise ResultError("response request_id does not match the evaluation request")

    try:
        status = RunStatus(str(value.get("status")))
    except ValueError as error:
        raise ResultError(f"invalid response status: {value.get('status')!r}") from error

    metrics_value = value.get("metrics", {})
    if not isinstance(metrics_value, Mapping):
        raise ResultError("metrics must be an object")
    metrics: dict[str, float] = {}
    for key, metric_value in metrics_value.items():
        if not isinstance(key, str) or not key:
            raise ResultError("metric names must be non-empty strings")
        metrics[key] = _finite_float(metric_value, f"metric {key!r}")
    if status is RunStatus.OK:
        missing = [metric for metric in required_metrics if metric not in metrics]
        if missing:
            raise ResultError(f"successful response is missing metrics: {', '.join(missing)}")
        if "qps" in metrics and metrics["qps"] <= 0:
            raise ResultError("qps must be positive")
        if "recall" in metrics and not 0.0 <= metrics["recall"] <= 1.0:
            raise ResultError("recall must be in [0, 1]")

    auxiliary = value.get("auxiliary", {})
    if not isinstance(auxiliary, Mapping):
        raise ResultError("auxiliary must be an object")
    provenance = value.get("provenance")
    merged_auxiliary = dict(auxiliary)
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise ResultError("provenance must be an object")
        merged_auxiliary["provenance"] = dict(provenance)

    artifacts = _response_artifacts(value.get("artifacts", []), run_dir)
    error_value = value.get("error")
    if error_value is not None and not isinstance(error_value, str):
        raise ResultError("error must be a string or null")
    return (
        Observation(
            status=status,
            metrics=metrics,
            auxiliary=merged_auxiliary,
            error=error_value,
        ),
        artifacts,
    )


def _response_artifacts(value: Any, run_dir: Path) -> list[str]:
    if not isinstance(value, list):
        raise ResultError("artifacts must be a list")
    artifacts: list[str] = []
    for entry in value:
        checksum: str | None = None
        if isinstance(entry, str):
            relative = entry
        elif isinstance(entry, Mapping):
            relative = entry.get("path")
            checksum_value = entry.get("sha256")
            if checksum_value is not None and not isinstance(checksum_value, str):
                raise ResultError("artifact sha256 must be a string")
            checksum = checksum_value
        else:
            raise ResultError("artifact entries must be strings or objects")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ResultError("artifact paths must be non-empty relative paths")
        try:
            artifact_path = ensure_within(run_dir / relative, run_dir)
        except ValueError as error:
            raise ResultError(str(error)) from error
        if not artifact_path.is_file():
            raise ResultError(f"plugin artifact does not exist: {relative}")
        if checksum is not None:
            actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual != checksum:
                raise ResultError(f"plugin artifact checksum mismatch: {relative}")
        artifacts.append(str(artifact_path))
    return artifacts


def _validate_command(value: Sequence[Any], roots_value: Any) -> tuple[str, ...]:
    if not all(isinstance(item, str) and item and "\x00" not in item for item in value):
        raise ConfigurationError("subprocess command must contain non-empty strings without NUL")
    command = list(value)
    executable_value = command[0]
    if os.path.isabs(executable_value) or os.sep in executable_value:
        executable = Path(executable_value).expanduser().resolve()
        if not executable.is_file():
            raise ConfigurationError(f"subprocess executable does not exist: {executable}")
    else:
        located = shutil.which(executable_value)
        if located is None:
            raise ConfigurationError(f"subprocess executable is not on PATH: {executable_value}")
        executable = Path(located).resolve()

    if roots_value is not None:
        if not isinstance(roots_value, (list, tuple)) or not roots_value:
            raise ConfigurationError("executable_roots must be a non-empty path list")
        allowed = False
        for root_value in roots_value:
            if not isinstance(root_value, str) or not root_value:
                raise ConfigurationError("executable_roots entries must be strings")
            root = Path(root_value).expanduser().resolve()
            try:
                ensure_within(executable, root)
                allowed = True
                break
            except ValueError:
                continue
        if not allowed:
            raise ConfigurationError(
                f"subprocess executable is outside executable_roots: {executable}"
            )
    command[0] = str(executable)
    return tuple(command)


def _sanitized_environment(pass_env: Sequence[str]) -> dict[str, str]:
    allowed_names = {"LANG", "LC_ALL", "LC_CTYPE", "PATH", "SYSTEMROOT", "TMPDIR"}
    allowed_names.update(pass_env)
    return {name: os.environ[name] for name in allowed_names if name in os.environ}


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


__all__ = [
    "OneShotSubprocessRunner",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "SubprocessPluginRunner",
]
