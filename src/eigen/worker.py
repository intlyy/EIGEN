"""Optional SSH transport; reuse local execution on each MiniDB machine."""

from __future__ import annotations

import argparse
import contextlib
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field

from eigen.config import LoadedProject, StrictModel, load_project
from eigen.errors import EIGENError, RunnerError
from eigen.execution import evaluate_candidates, stage_project, tune_project
from eigen.profiles import profile_fingerprint
from eigen.utils import atomic_write_json, dataset_sha256, fingerprint


class RemoteWorker(StrictModel):
    # SSH config aliases also support custom ports, keys and jump hosts.
    host: str = Field(pattern=r"^(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*$")
    project_config: str = Field(min_length=1)
    python_executable: str = Field(default="python3", min_length=1)
    timeout_s: float = Field(default=604800, gt=0)


class WorkerRequest(StrictModel):
    schema_version: Literal[1] = 1
    stage: Literal["tuning", "validation"]
    run_id: str = Field(pattern=r"^[a-f0-9]{16}$")
    expected_contract: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_dataset_sha256: str | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class WorkerResponse(StrictModel):
    schema_version: Literal[1] = 1
    artifact_dir: str
    worker_wall_s: float = Field(ge=0)
    result: dict[str, Any] | list[dict[str, Any]]


def worker_contract(project: LoadedProject) -> str:
    """Match experiment semantics while allowing machine-specific locations."""
    payload = project.config.model_dump(mode="json")
    for key in ("engine_profile", "artifact_dir"):
        payload.pop(key)
    for key in ("host", "connection_params"):
        payload["execution"].pop(key)
    for key in ("repo_path", "python_executable", "dataset_path", "dataset_cache"):
        payload["runner"]["settings"].pop(key, None)
    settings = payload["lifecycle"]["settings"]
    for key in ("compose_file", "project_name", "endpoint", "ready_check"):
        settings.pop(key, None)
    for key in ("EIGEN_PORT", "EIGEN_GRPC_PORT", "EIGEN_HTTP_PORT"):
        settings.get("environment", {}).pop(key, None)
    if not settings.get("environment"):
        settings.pop("environment", None)
    payload["profile"] = profile_fingerprint(project.profile)
    return fingerprint(payload, length=64)


def _run_command(argv: list[str], timeout_s: float, *, input_text: str | None = None):
    try:
        process = subprocess.run(
            argv,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RunnerError(f"remote worker {argv[0]} failed: {error}") from error
    if process.returncode:
        raise RunnerError(f"remote worker {argv[0]} failed: {process.stderr[-2000:]}")
    return process


def run_remote_worker(
    worker: RemoteWorker,
    project: LoadedProject,
    artifact_dir: Path,
    *,
    stage: Literal["tuning", "validation"],
    run_id: str,
    candidates: list[dict[str, Any]] | None = None,
) -> tuple[Any, float]:
    request = WorkerRequest(
        stage=stage,
        run_id=run_id,
        expected_contract=worker_contract(project),
        expected_dataset_sha256=project.config.runner.settings.get("expected_dataset_sha256"),
        candidates=candidates or [],
    )
    command = shlex.join([worker.python_executable, "-m", "eigen.worker", worker.project_config])
    process = _run_command(
        ["ssh", "-T", "-o", "BatchMode=yes", "--", worker.host, command],
        worker.timeout_s,
        input_text=request.model_dump_json(),
    )
    response = WorkerResponse.model_validate_json(process.stdout)
    if not isinstance(response.result, dict if stage == "tuning" else list):
        raise RunnerError("remote worker returned the wrong result type for its stage")
    remote_dir = PurePosixPath(response.artifact_dir)
    if (
        not remote_dir.is_absolute()
        or remote_dir == PurePosixPath("/")
        or re.search(r"[\r\n\x00]", response.artifact_dir)
    ):
        raise RunnerError("remote worker returned an invalid artifact directory")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # SFTP-based scp (OpenSSH >= 9) treats the remote path as a literal path.
    _run_command(
        [
            "scp",
            "-r",
            "-o",
            "BatchMode=yes",
            "--",
            f"{worker.host}:{remote_dir}/.",
            str(artifact_dir.resolve()),
        ],
        worker.timeout_s,
    )
    return response.result, response.worker_wall_s


def execute_worker(project: LoadedProject, request: WorkerRequest) -> WorkerResponse:
    settings = project.config.runner.settings
    if request.expected_dataset_sha256 is not None:
        settings["expected_dataset_sha256"] = request.expected_dataset_sha256
    if worker_contract(project) != request.expected_contract:
        raise ValueError("remote project differs from the coordinator's experiment contract")
    if request.expected_dataset_sha256 is not None:
        dataset = settings.get("dataset_path")
        if dataset is None or dataset_sha256(dataset) != request.expected_dataset_sha256:
            raise ValueError("remote MiniDB content differs from its manifest checksum")
    directory = project.artifact_dir / "studies" / request.run_id / request.stage
    staged = stage_project(project, directory, request.stage)
    started = time.monotonic()
    if request.stage == "tuning":
        result = tune_project(staged)
    else:
        result = evaluate_candidates(staged, request.candidates)
    elapsed = time.monotonic() - started
    atomic_write_json(directory / "result.json", result)
    return WorkerResponse(artifact_dir=str(directory), worker_wall_s=elapsed, result=result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        request = WorkerRequest.model_validate_json(sys.stdin.read())
        # Keep stdout exclusively for the protocol, including with custom runners.
        with contextlib.redirect_stdout(sys.stderr):
            response = execute_worker(load_project(args.config), request)
        sys.stdout.reconfigure(encoding="utf-8")
        print(response.model_dump_json())
        return 0
    except (EIGENError, OSError, TypeError, ValueError) as error:
        print(f"eigen worker: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
