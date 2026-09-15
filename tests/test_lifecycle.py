from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mutune.errors import ConfigurationError, RunnerError
from mutune.lifecycle import (
    DockerComposeLifecycle,
    ExternalLifecycle,
    ReadyCheck,
    create_lifecycle,
)


class LifecycleTests(unittest.TestCase):
    def test_lifecycle_defaults_to_external(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lifecycle = create_lifecycle(
                None,
                workspace_root=root,
                artifact_dir=root / "artifacts",
                default_endpoint="localhost",
            )
            self.assertIsInstance(lifecycle, ExternalLifecycle)
            self.assertEqual(lifecycle.start(), "localhost")
            lifecycle.stop()

    def test_external_lifecycle_rejects_server_tuning(self) -> None:
        lifecycle = ExternalLifecycle("localhost", ReadyCheck(kind="none"))
        with self.assertRaisesRegex(ConfigurationError, "cannot apply"):
            lifecycle.configure(
                "pgvector",
                {"postgresql": {"shared_buffers_mb": 12288}},
            )

    def test_tcp_ready_check_connects_to_listening_socket(self) -> None:
        class Connected:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        calls = []

        def connect(address, timeout):
            calls.append((address, timeout))
            return Connected()

        check = ReadyCheck(
            kind="tcp",
            host="127.0.0.1",
            port=6333,
            timeout_s=1.0,
            interval_s=0.01,
            connect_timeout_s=0.1,
        )
        lifecycle = ExternalLifecycle("127.0.0.1:6333", check)
        with mock.patch.object(socket, "create_connection", side_effect=connect):
            self.assertEqual(lifecycle.start(), "127.0.0.1:6333")
        self.assertEqual(calls, [(("127.0.0.1", 6333), 0.1)])

    def test_ready_check_timeout_is_bounded(self) -> None:
        check = ReadyCheck(
            kind="tcp",
            host="127.0.0.1",
            port=9,
            timeout_s=0.03,
            interval_s=0.005,
            connect_timeout_s=0.005,
        )
        with mock.patch.object(socket, "create_connection", side_effect=OSError("refused")):
            with self.assertRaisesRegex(RunnerError, "did not become ready"):
                check.wait()

    def test_compose_file_must_stay_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside-compose.yaml"
            outside.write_text("services: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes allowed root"):
                DockerComposeLifecycle(
                    endpoint="localhost:1234",
                    ready_check=ReadyCheck(kind="none"),
                    compose_file=outside,
                    workspace_root=workspace,
                    project_name="mutune-test",
                    artifact_dir=root / "artifacts",
                )

    def test_compose_lifecycle_uses_argv_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compose = root / "compose.yaml"
            compose.write_text("services: {}\n", encoding="utf-8")
            calls: list[tuple[list[str], dict]] = []

            def fake_run(argv, **kwargs):
                calls.append((argv, kwargs))
                return subprocess.CompletedProcess(argv, 0, "ok", "")

            lifecycle = DockerComposeLifecycle(
                endpoint="localhost:1234",
                ready_check=ReadyCheck(kind="none"),
                compose_file=compose,
                workspace_root=root,
                project_name="mutune-test",
                artifact_dir=root / "artifacts",
            )
            with mock.patch.object(subprocess, "run", side_effect=fake_run):
                lifecycle.start()
                lifecycle.stop()

            self.assertEqual(calls[0][0][:2], ["docker", "compose"])
            self.assertIs(calls[0][1]["shell"], False)
            self.assertTrue(any("up" in argv for argv, _kwargs in calls))
            self.assertTrue(any("down" in argv for argv, _kwargs in calls))

    def test_compose_restart_can_preserve_database_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compose = root / "compose.yaml"
            compose.write_text("services: {}\n", encoding="utf-8")
            calls: list[list[str]] = []

            def fake_run(argv, **kwargs):
                calls.append(argv)
                return subprocess.CompletedProcess(argv, 0, "ok", "")

            lifecycle = DockerComposeLifecycle(
                endpoint="localhost:5432",
                ready_check=ReadyCheck(kind="none"),
                compose_file=compose,
                workspace_root=root,
                project_name="mutune-preserve-test",
                artifact_dir=root / "artifacts",
                remove_volumes=True,
            )
            with mock.patch.object(subprocess, "run", side_effect=fake_run):
                lifecycle.start()
                lifecycle.restart(preserve_data=True)
                lifecycle.stop()

            force_recreate = [argv for argv in calls if "--force-recreate" in argv]
            self.assertEqual(len(force_recreate), 1)
            self.assertIn("up", force_recreate[0])
            down_calls = [argv for argv in calls if "down" in argv]
            self.assertEqual(len(down_calls), 1)
            self.assertIn("--volumes", down_calls[0])

    def test_compose_rejects_environment_that_changes_docker_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compose = root / "compose.yaml"
            compose.write_text("services: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "DOCKER_HOST"):
                DockerComposeLifecycle(
                    endpoint="localhost:1234",
                    ready_check=ReadyCheck(kind="none"),
                    compose_file=compose,
                    workspace_root=root,
                    project_name="mutune-test",
                    artifact_dir=root / "artifacts",
                    environment={"DOCKER_HOST": "ssh://unexpected"},
                )

    def test_postgresql_configuration_is_typed_rendered_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compose = root / "compose.yaml"
            compose.write_text("services:\n  pgvector:\n    image: pgvector\n")
            artifacts = root / "artifacts"
            lifecycle = DockerComposeLifecycle(
                endpoint="localhost:5432",
                ready_check=ReadyCheck(kind="none"),
                compose_file=compose,
                workspace_root=root,
                project_name="mutune-postgres-test",
                artifact_dir=artifacts,
                server_config={
                    "kind": "postgresql",
                    "service": "pgvector",
                    "user": "postgres",
                    "database": "postgres",
                },
            )
            params = {
                "postgresql": {
                    "shared_buffers_mb": 12288,
                    "max_parallel_workers": 16,
                    "max_parallel_maintenance_workers": 15,
                    "random_page_cost": 1.1,
                }
            }
            self.assertTrue(lifecycle.configure("pgvector", params))
            self.assertFalse(lifecycle.configure("pgvector", params))

            override = json.loads((artifacts / "server-compose.override.json").read_text())
            command = override["services"]["pgvector"]["command"]
            self.assertEqual(command[0], "postgres")
            self.assertIn("shared_buffers=12288MB", command)
            self.assertIn("max_parallel_workers=16", command)
            self.assertIn("max_parallel_maintenance_workers=15", command)
            self.assertIn("random_page_cost=1.1", command)
            self.assertIn(
                str(artifacts / "server-compose.override.json"),
                lifecycle._argv("up", "-d"),
            )

            actual = {
                "shared_buffers_mb": 12288 * 1024 * 1024,
                "max_parallel_workers": 16,
                "max_parallel_maintenance_workers": 15,
                "random_page_cost": 1.1,
            }

            def fake_run(argv, **kwargs):
                stdout = json.dumps(actual) if "exec" in argv else "ok"
                return subprocess.CompletedProcess(argv, 0, stdout, "")

            with mock.patch.object(subprocess, "run", side_effect=fake_run):
                lifecycle.start()
                lifecycle.stop()

            self.assertEqual(
                json.loads((artifacts / "effective-server-config.json").read_text()),
                actual,
            )
            ready = json.loads((artifacts / "postgres-ready.json").read_text())
            self.assertEqual(ready["attempts"], 1)

    def test_postgresql_readiness_retries_inside_container(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compose = root / "compose.yaml"
            compose.write_text("services:\n  pgvector:\n    image: pgvector\n")
            artifacts = root / "artifacts"
            lifecycle = DockerComposeLifecycle(
                endpoint="localhost:5432",
                ready_check=ReadyCheck(
                    kind="none",
                    timeout_s=1,
                    interval_s=0.001,
                    connect_timeout_s=0.1,
                ),
                compose_file=compose,
                workspace_root=root,
                project_name="mutune-postgres-retry-test",
                artifact_dir=artifacts,
                server_config={"kind": "postgresql", "service": "pgvector"},
            )
            lifecycle.configure(
                "pgvector",
                {"postgresql": {"shared_buffers_mb": 12288}},
            )
            readiness_attempts = 0

            def fake_run(argv, **kwargs):
                nonlocal readiness_attempts
                if "pg_isready" in argv:
                    readiness_attempts += 1
                    if readiness_attempts < 3:
                        return subprocess.CompletedProcess(
                            argv,
                            2,
                            "",
                            "postgres:5432 - no response",
                        )
                    return subprocess.CompletedProcess(
                        argv,
                        0,
                        "postgres:5432 - accepting connections",
                        "",
                    )
                if "psql" in argv:
                    payload = {"shared_buffers_mb": 12288 * 1024 * 1024}
                    return subprocess.CompletedProcess(
                        argv,
                        0,
                        json.dumps(payload),
                        "",
                    )
                return subprocess.CompletedProcess(argv, 0, "ok", "")

            with mock.patch.object(subprocess, "run", side_effect=fake_run):
                lifecycle.start()
                lifecycle.stop()

            self.assertEqual(readiness_attempts, 3)
            ready = json.loads((artifacts / "postgres-ready.json").read_text())
            self.assertEqual(ready["attempts"], 3)

    def test_postgresql_configuration_rejects_unknown_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compose = root / "compose.yaml"
            compose.write_text("services: {}\n")
            lifecycle = DockerComposeLifecycle(
                endpoint="localhost:5432",
                ready_check=ReadyCheck(kind="none"),
                compose_file=compose,
                workspace_root=root,
                project_name="mutune-postgres-test",
                artifact_dir=root / "artifacts",
                server_config={"kind": "postgresql", "service": "pgvector"},
            )
            with self.assertRaisesRegex(ConfigurationError, "unsupported PostgreSQL"):
                lifecycle.configure(
                    "pgvector",
                    {"postgresql": {"arbitrary_setting": "unsafe"}},
                )

    def test_milvus_configuration_mounts_complete_versioned_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compose = root / "compose.yaml"
            compose.write_text("services:\n  standalone:\n    image: milvus\n")
            artifacts = root / "artifacts"
            lifecycle = DockerComposeLifecycle(
                endpoint="localhost:19530",
                ready_check=ReadyCheck(kind="none"),
                compose_file=compose,
                workspace_root=root,
                project_name="mutune-milvus-test",
                artifact_dir=artifacts,
                server_config={"kind": "milvus_user_yaml", "service": "standalone"},
            )
            params = {
                "milvus": {
                    "data_coord_segment_max_size_mb": 512,
                    "data_coord_segment_seal_proportion": 0.23,
                    "query_coord_auto_handoff": True,
                    "query_coord_auto_balance": True,
                    "common_graceful_time_ms": 5000,
                    "data_node_segment_insert_buf_size_bytes": 16777216,
                    "root_coord_min_segment_size_to_enable_index": 1024,
                }
            }

            self.assertTrue(lifecycle.configure("milvus", params))

            yaml = (artifacts / "milvus.yaml").read_text()
            config = json.loads(yaml)  # JSON is a YAML subset accepted by Milvus.
            self.assertEqual(config["dataCoord"]["segment"]["maxSize"], 512)
            self.assertEqual(config["dataCoord"]["segment"]["sealProportion"], 0.23)
            self.assertIs(config["queryCoord"]["autoHandoff"], True)
            self.assertEqual(config["common"]["gracefulTime"], 5000)
            self.assertEqual(config["dataNode"]["segment"]["insertBufSize"], 16777216)
            for section in ("etcd", "minio", "queryNode", "rootCoord"):
                self.assertIn(section, config)  # Partial replacement would erase defaults.
            override = json.loads((artifacts / "server-compose.override.json").read_text())
            mount = override["services"]["standalone"]["volumes"][0]
            self.assertEqual(mount["target"], "/milvus/configs/milvus.yaml")
            self.assertTrue(mount["read_only"])
            with mock.patch.object(
                lifecycle, "_run", return_value=subprocess.CompletedProcess([], 0, yaml, "")
            ) as run:
                lifecycle._verify_milvus_yaml()
            self.assertEqual(run.call_args.args[-1], "/milvus/configs/milvus.yaml")
            verification = json.loads((artifacts / "milvus-config-verification.json").read_text())
            self.assertFalse(verification["runtime_values_verified"])
            with mock.patch.object(
                lifecycle, "_run", return_value=subprocess.CompletedProcess([], 0, "wrong", "")
            ):
                with self.assertRaisesRegex(RunnerError, "not mounted"):
                    lifecycle._verify_milvus_yaml()


if __name__ == "__main__":
    unittest.main()
