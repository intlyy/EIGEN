"""Generated studies retain the requested budget and explicit geo execution contract."""

from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from eigen.benchmark_compat import MILVUS_GEO_CONTRACT
from eigen.geo_minidb import build_geo_minidbs
from eigen.lifecycle import DockerComposeLifecycle, create_lifecycle
from eigen.minidb import build_hdf5_minidbs
from eigen.study import load_study
from scripts.prepare_reproduction import main as prepare_reproduction
from scripts.prepare_study import main


class StudyGenerationTests(unittest.TestCase):
    def _source(self, root):
        source = root / "source"
        source.mkdir()
        np.save(source / "vectors.npy", np.ones((30, 2), dtype=np.float32))
        (source / "payloads.jsonl").write_text(
            "".join(json.dumps({"location": {"lat": i, "lon": 0}}) + "\n" for i in range(30)),
            encoding="utf-8",
        )
        query = {
            "query": [1, 1],
            "conditions": {"and": [{"location": {"geo": {"lat": 0, "lon": 0, "radius": 1}}}]},
            "closest_ids": [0],
        }
        (source / "tests.jsonl").write_text(json.dumps(query) + "\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            build_geo_minidbs(source, root / "minis", top_k=1)
        benchmark = root / "benchmark"
        benchmark.mkdir()
        (benchmark / "run.py").write_text("# fingerprint fixture; never executed\n")
        return root / "minis/manifest.json", benchmark

    def _generate(self, engine, manifest, benchmark, output, *extra):
        arguments = [
            "prepare_study.py",
            "--engine",
            engine,
            "--manifest",
            str(manifest),
            "--benchmark-repo",
            str(benchmark),
            "--output",
            str(output),
            "--top-k",
            "1",
            *extra,
        ]
        with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            main()

    def _assert_compose_projects(self, study):
        targets = {
            "milvus": {"EIGEN_PORT": 19530, "EIGEN_HTTP_PORT": 9091},
            "qdrant": {"EIGEN_PORT": 6333, "EIGEN_GRPC_PORT": 6334},
            "pgvector": {"EIGEN_PORT": 5432},
        }
        for project in [*study.minidbs, study.full_database]:
            with self.subTest(project=project.source_path):
                settings = project.config.lifecycle.settings
                with (
                    patch("eigen.lifecycle.subprocess.run") as process,
                    patch("eigen.lifecycle.socket.create_connection") as connect,
                ):
                    lifecycle = create_lifecycle(
                        project.config.lifecycle.model_dump(mode="json"),
                        workspace_root=project.source_path.parent,
                        artifact_dir=project.artifact_dir / "lifecycle",
                        default_endpoint=project.config.execution.host,
                    )
                    process.assert_not_called()
                    connect.assert_not_called()
                self.assertIsInstance(lifecycle, DockerComposeLifecycle)
                self.assertTrue(lifecycle.project_name.startswith("eigen-"))
                self.assertIn(lifecycle.project_name, lifecycle._argv("config"))
                self.assertFalse(lifecycle.artifact_dir.exists())
                engine = project.profile.adapter.engine
                environment = lifecycle.environment
                self.assertEqual(set(environment), set(targets[engine]))
                template = lifecycle.compose_file.read_text(encoding="utf-8")
                variables = set(re.findall(r"\$\{(EIGEN_[A-Z_]+):", template))
                self.assertEqual(variables, set(targets[engine]) | {"EIGEN_CPUS", "EIGEN_MEMORY"})
                rendered = re.sub(
                    r"\$\{([A-Z_]+):-([^}]*)\}",
                    lambda match, values=environment: values.get(match[1], match[2]),
                    template,
                )
                mappings = {
                    int(target): int(host)
                    for host, target in re.findall(r'"127\.0\.0\.1:(\d+):(\d+)"', rendered)
                }
                self.assertEqual(
                    mappings,
                    {target: int(environment[name]) for name, target in targets[engine].items()},
                )
                port = int(environment["EIGEN_PORT"])
                self.assertEqual(settings["ready_check"]["port"], port)
                connection = project.config.execution.connection_params
                if engine == "qdrant":
                    self.assertEqual(project.config.execution.host, f"http://127.0.0.1:{port}")
                    self.assertEqual(connection["grpc_port"], int(environment["EIGEN_GRPC_PORT"]))
                else:
                    self.assertEqual(connection["port"], port)

    def test_both_generators_create_eigen_lifecycles_and_matching_compose_ports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.hdf5"
            vectors = np.random.default_rng(12).normal(size=(30, 4)).astype(np.float32)
            with h5py.File(source, "w") as handle:
                handle["train"], handle["test"] = vectors, vectors[:2]
            build_hdf5_minidbs(source, root / "minis", top_k=1)
            manifest = root / "minis/manifest.json"
            benchmark = root / "benchmark"
            benchmark.mkdir()
            (benchmark / "run.py").write_text("# offline fingerprint fixture\n", encoding="utf-8")
            for engine in ("milvus", "qdrant", "pgvector"):
                output = root / engine
                self._generate(engine, manifest, benchmark, output)
                self._assert_compose_projects(load_study(output / "study.json"))
            inventory = root / "inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "fixture",
                                "description": "Offline lifecycle integration fixture",
                                "manifests": [{"label": "m3-r10", "path": str(manifest)}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_reproduction(
                    [
                        "--inventory",
                        str(inventory),
                        "--benchmark-repo",
                        str(benchmark),
                        "--output",
                        str(root / "matrix"),
                        "--engines",
                        "milvus",
                        "qdrant",
                        "pgvector",
                        "--recalls",
                        "0.95",
                        "--top-k",
                        "1",
                    ]
                )
            plan = json.loads((root / "matrix/run-plan.json").read_text(encoding="utf-8"))
            self.assertEqual(len(plan["runs"]), 3)
            for run in plan["runs"]:
                self._assert_compose_projects(load_study(run["config"]))

    def test_milvus_and_qdrant_geo_studies_use_three_twenty_evaluation_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, benchmark = self._source(root)
            for engine in ("milvus", "qdrant"):
                output = root / engine
                self._generate(engine, manifest, benchmark, output)
                study = load_study(output / "study.json")
                self.assertEqual(len(study.minidbs), 3)
                self.assertEqual([p.config.tuning.budget for p in study.minidbs], [20, 20, 20])
                self.assertEqual(sum(p.config.tuning.budget for p in study.minidbs), 60)
                self.assertEqual(len({p.config.tuning.seed for p in study.minidbs}), 3)
                for project in [*study.minidbs, study.full_database]:
                    self.assertIsNone(project.config.tuning.initial_samples)
                    self.assertTrue(project.config.execution.filtered)
                    settings = project.config.runner.settings
                    self.assertEqual(settings["dataset_entry"]["schema"], {"location": "geo"})
                    self.assertIn("expected_source_sha256", settings)
                    self.assertEqual(
                        settings.get("milvus_geo_filter"),
                        MILVUS_GEO_CONTRACT if engine == "milvus" else None,
                    )

    def test_milvus_geo_cannot_silently_fall_back_to_unadapted_upstream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, benchmark = self._source(root)
            output = root / "milvus"
            self._generate("milvus", manifest, benchmark, output)
            path = output / "milvus-mini-01.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            del config["runner"]["settings"]["milvus_geo_filter"]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicit payload-ID"):
                load_study(output / "study.json")

    def test_geo_pgvector_is_rejected_and_budget_override_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, benchmark = self._source(root)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self._generate("pgvector", manifest, benchmark, root / "pgvector")
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((root / "pgvector").exists())
            self._generate(
                "milvus", manifest, benchmark, root / "custom", "--budget-per-minidb", "21"
            )
            study = load_study(root / "custom/study.json")
            self.assertEqual([p.config.tuning.budget for p in study.minidbs], [21, 21, 21])

    def test_experiment_matrix_controls_reach_every_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, benchmark = self._source(root)
            self._generate(
                "milvus",
                manifest,
                benchmark,
                root / "matrix",
                "--recall-threshold",
                "0.975",
                "--seed",
                "101",
                "--dataset-description",
                "Geo-radius experiment fixture",
                "--search-parallel",
                "8",
                "--upload-parallel",
                "4",
            )
            study = load_study(root / "matrix/study.json")
            projects = [*study.minidbs, study.full_database]
            self.assertEqual([p.config.tuning.seed for p in projects], [101, 102, 103, 104])
            for project in projects:
                self.assertEqual(project.config.tuning.recall_threshold, 0.975)
                self.assertEqual(project.config.execution.search_parallel, 8)
                self.assertEqual(project.config.execution.upload_parallel, 4)
                self.assertEqual(
                    project.config.execution.dataset_description, "Geo-radius experiment fixture"
                )

    def test_invalid_matrix_controls_fail_before_writing_configs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, benchmark = self._source(root)
            for index, arguments in enumerate(
                (
                    ("--recall-threshold", "nan"),
                    ("--recall-threshold", "1.1"),
                    ("--search-parallel", "0"),
                    ("--upload-parallel", "-1"),
                )
            ):
                output = root / str(index)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self._generate("milvus", manifest, benchmark, output, *arguments)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
