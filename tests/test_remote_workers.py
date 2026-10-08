"""Offline remote-worker transport, isolation and paper timing regressions."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from eigen.config import load_project
from eigen.errors import RunnerError
from eigen.study import LoadedStudy, StudyConfig, load_study, run_study
from eigen.utils import dataset_sha256
from eigen.worker import (
    RemoteWorker,
    WorkerRequest,
    WorkerResponse,
    execute_worker,
    run_remote_worker,
    worker_contract,
)


def project(root, index=0):
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data.bin"
    data.write_bytes(b"same MiniDB content")
    path = root / "project with spaces.json"
    path.write_text(
        json.dumps(
            {
                "experiment_name": f"mini-{index}",
                "engine_profile": "qdrant-hnsw-dense",
                "artifact_dir": "./artifacts",
                "execution": {"dataset": str(index), "distance": "l2", "vector_size": 4},
                "runner": {
                    "plugin": "dry-run",
                    "settings": {
                        "dataset_path": str(data),
                        "expected_dataset_sha256": dataset_sha256(data),
                    },
                },
                "tuning": {
                    "strategy": "random",
                    "budget": 2,
                    "initial_samples": 2,
                    "seed": 42 + index,
                },
            }
        ),
        encoding="utf-8",
    )
    return load_project(path)


def study_config(**kwargs):
    return StudyConfig(
        minidbs=["mini-0.json", "mini-1.json"],
        full_database="full.json",
        minidb_manifest="manifest.json",
        **kwargs,
    )


class RemoteWorkerTests(unittest.TestCase):
    def test_worker_command_line_reads_request_and_emits_only_protocol_json(self):
        with tempfile.TemporaryDirectory() as directory:
            loaded = project(Path(directory))
            request = WorkerRequest(
                stage="tuning",
                run_id="a" * 16,
                expected_contract=worker_contract(loaded),
                expected_dataset_sha256=loaded.config.runner.settings["expected_dataset_sha256"],
            )
            code = (
                f"import sys, runpy; sys.path[:0] = {sys.path!r}; "
                "runpy.run_module('eigen.worker', run_name='__main__', alter_sys=True)"
            )
            process = subprocess.run(
                [sys.executable, "-c", code, str(loaded.source_path)],
                input=request.model_dump_json(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=20,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            response = WorkerResponse.model_validate_json(process.stdout)
            self.assertTrue(response.result["complete"])
            self.assertTrue((Path(response.artifact_dir) / "history.jsonl").is_file())

    def test_remote_workers_require_one_separate_host_per_view(self):
        worker = {"host": "user@worker-a", "project_config": "/srv/mini.json"}
        for workers in ([], [worker], [worker, {**worker, "host": "other@WORKER-A"}]):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                study_config(remote_workers=workers)
        for host in ("-oProxyCommand=bad", "host; bad", "host name"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                RemoteWorker(host=host, project_config="project.json")

    def test_loopback_endpoints_are_scoped_to_each_remote_machine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projects = [project(root / str(i), i) for i in range(3)]
            manifest = {
                "bucket_fingerprint": "fixture",
                "dimension": 4,
                "metric": "l2",
                "top_k": 10,
                "source": projects[2].config.runner.settings["dataset_path"],
                "source_sha256": projects[2].config.runner.settings["expected_dataset_sha256"],
                "minidbs": [
                    {
                        "path": p.config.runner.settings["dataset_path"],
                        "sample_seed": i,
                        "sha256": p.config.runner.settings["expected_dataset_sha256"],
                    }
                    for i, p in enumerate(projects[:2])
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            config = {
                "minidbs": [str(p.source_path) for p in projects[:2]],
                "full_database": str(projects[2].source_path),
                "minidb_manifest": "manifest.json",
            }
            path = root / "study.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "distinct database endpoints"):
                load_study(path)
            config["remote_workers"] = [
                {
                    "host": f"worker-{i}",
                    "project_config": "/srv/mini.json",
                }
                for i in range(2)
            ]
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(len(load_study(path).minidbs), 2)

    def test_real_worker_tuning_validation_and_resume_over_mock_ssh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coordinator = project(root / "coordinator")
            remote = project(root / "remote")
            remote.config.execution.host = "remote-localhost"
            self.assertEqual(worker_contract(coordinator), worker_contract(remote))
            worker = RemoteWorker(host="worker-a", project_config="/srv/project with spaces.json")
            requests = []

            def transport(argv, **kwargs):
                if argv[0] == "ssh":
                    self.assertIn("BatchMode=yes", argv)
                    self.assertIn("'/srv/project with spaces.json'", argv[-1])
                    request = WorkerRequest.model_validate_json(kwargs["input"])
                    requests.append(request)
                    response = execute_worker(remote, request)
                    # Represent the Windows fixture using a POSIX remote path.
                    response.artifact_dir = "/srv/artifacts/" + request.stage
                    return subprocess.CompletedProcess(argv, 0, response.model_dump_json(), "")
                self.assertEqual(argv[0], "scp")
                request = requests[-1]
                source = remote.artifact_dir / "studies" / request.run_id / request.stage
                shutil.copytree(source, Path(argv[-1]), dirs_exist_ok=True)
                return subprocess.CompletedProcess(argv, 0, "", "")

            with patch("eigen.worker.subprocess.run", side_effect=transport):
                result, elapsed = run_remote_worker(
                    worker,
                    coordinator,
                    root / "collected/tuning",
                    stage="tuning",
                    run_id="a" * 16,
                )
                self.assertTrue(result["complete"])
                self.assertEqual(result["resumed_evaluations"], 0)
                self.assertGreater(elapsed, 0)
                self.assertTrue((root / "collected/tuning/history.jsonl").is_file())
                resumed, _ = run_remote_worker(
                    worker,
                    coordinator,
                    root / "collected/tuning",
                    stage="tuning",
                    run_id="a" * 16,
                )
                self.assertEqual(resumed["resumed_evaluations"], 2)
                measurements, _ = run_remote_worker(
                    worker,
                    coordinator,
                    root / "collected/validation",
                    stage="validation",
                    run_id="a" * 16,
                    candidates=[{}],
                )
                self.assertEqual(len(measurements), 1)
                self.assertEqual(measurements[0]["status"], "ok")
                self.assertTrue((root / "collected/validation/measurements/00000.json").is_file())

    def test_wrong_remote_contract_or_data_fails_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, remote = project(root / "local"), project(root / "remote")
            request = WorkerRequest(
                stage="tuning",
                run_id="a" * 16,
                expected_contract=worker_contract(local),
                expected_dataset_sha256=local.config.runner.settings["expected_dataset_sha256"],
            )
            remote.config.tuning.budget = 3
            with patch("eigen.worker.tune_project") as tune:
                with self.assertRaisesRegex(ValueError, "experiment contract"):
                    execute_worker(remote, request)
                remote.config.tuning.budget = 2
                Path(remote.config.runner.settings["dataset_path"]).write_bytes(b"wrong data")
                with self.assertRaisesRegex(ValueError, "manifest checksum"):
                    execute_worker(remote, request)
                tune.assert_not_called()

    def test_ssh_failure_timeout_and_bad_protocol_do_not_become_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = project(root)
            worker = RemoteWorker(host="worker-a", project_config="/srv/mini.json")
            for failure in (
                subprocess.CompletedProcess([], 255, "", "connection failed"),
                subprocess.CompletedProcess([], 0, "not JSON", ""),
                subprocess.TimeoutExpired("ssh", 1),
            ):
                with self.subTest(failure=failure):
                    kwargs = (
                        {"side_effect": failure}
                        if isinstance(failure, Exception)
                        else {
                            "return_value": failure,
                        }
                    )
                    with patch("eigen.worker.subprocess.run", **kwargs) as process:
                        with self.assertRaises((RunnerError, ValueError)):
                            run_remote_worker(
                                worker,
                                local,
                                root / "collected",
                                stage="tuning",
                                run_id="a" * 16,
                            )
                        self.assertEqual(process.call_count, 1)

    def test_both_minidb_stages_are_remote_parallel_and_costs_use_worker_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projects = [project(root / str(i), i) for i in range(3)]
            cfg = study_config(
                remote_workers=[
                    {
                        "host": f"worker-{i}",
                        "project_config": f"/srv/{i}.json",
                    }
                    for i in range(2)
                ]
            )
            study = LoadedStudy(
                cfg,
                projects[:2],
                projects[2],
                root / "study",
                {
                    "construction_timing": {"wall_s": 2},
                },
            )
            barrier = threading.Barrier(2)
            calls = []

            def remote(worker, project, artifact_dir, *, stage, run_id, candidates=None):
                barrier.wait(timeout=5)
                i = int(project.config.execution.dataset)
                calls.append((stage, worker.host, i))
                if stage == "tuning":
                    return {
                        "complete": True,
                        "transfer_candidates": [{"candidate": {"x": i}}],
                        "llm_wall_s": 0,
                        "resumed_evaluations": 0,
                    }, 12.0
                return [
                    {
                        "candidate": c,
                        "status": "ok",
                        "metrics": {"qps": 100 + c["x"], "recall": 1},
                    }
                    for c in candidates
                ], 3.0

            def full(project, candidates):
                self.assertEqual(project.config.execution.dataset, "2")
                return [
                    {
                        "candidate": c,
                        "status": "ok",
                        "metrics": {"qps": 200 - c["x"], "recall": 1},
                    }
                    for c in candidates
                ]

            local_tune = Mock(side_effect=AssertionError("MiniDB tuning ran locally"))
            with patch("eigen.study.run_remote_worker", side_effect=remote):
                result = run_study(study, tune_fn=local_tune, evaluate_fn=full)
            local_tune.assert_not_called()
            self.assertEqual(
                sorted(calls),
                [
                    ("tuning", "worker-0", 0),
                    ("tuning", "worker-1", 1),
                    ("validation", "worker-0", 0),
                    ("validation", "worker-1", 1),
                ],
            )
            self.assertEqual(result["best_candidate"], {"x": 0})
            self.assertEqual(result["cross_validation_evaluations"], 4)
            self.assertEqual(result["timings"]["local_optimization_worker_sum_s"], 24)
            self.assertEqual(result["timings"]["cross_validation_worker_sum_s"], 6)
            self.assertEqual(result["timings"]["paper_components_s"]["minidb_tuning"], 30)


if __name__ == "__main__":
    unittest.main()
