"""Offline Milvus geo pipeline checks; mocked outputs are not performance evidence."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eigen.api import EvaluationRequest, RunnerContext, WorkloadSpec
from eigen.benchmark_compat import MILVUS_GEO_CONTRACT
from eigen.errors import ConfigurationError
from eigen.profiles import load_profile
from eigen.rendering import ExperimentRenderer
from eigen.runners.vectordb_benchmark import VectorDBBenchmarkRunner

CONNECT = """def connect(host, connection_params):
    return connections.connect(alias="default", host=host,
        port=str(connection_params.get("port", 19530)), **connection_params)
"""


def source_snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class MilvusGeoRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "benchmark"
        for name in (
            "benchmark",
            "engine/base_client",
            "engine/clients/milvus",
            "dataset_reader",
            "datasets",
        ):
            (self.source / name).mkdir(parents=True)
        (self.source / "datasets/datasets.json").write_text("[]")
        (self.source / "run.py").write_text("# Synthetic subprocess outputs only.\n")
        (self.source / "engine/base_client/search.py").write_text(
            "import time\nDEFAULT_TOP = 10\nclass BaseSearcher:\n"
            "    @classmethod\n    def _search_one(cls, query, top=None):\n        return -1, 0\n"
        )
        for name, body in {
            "configure": "class MilvusConfigurator:\n"
            "    def recreate(self, dataset, collection_params):\n        return dataset\n",
            "search": "class MilvusSearcher:\n    @classmethod\n"
            "    def search_one(cls, query, top):\n        return []\n",
            "upload": 'UPLOADER_SENTINEL = "preserve ANN index construction"\n',
        }.items():
            (self.source / f"engine/clients/milvus/{name}.py").write_text(CONNECT + body)
        self.original_source = source_snapshot(self.source)
        self.dataset = self.root / "geo"
        self.dataset.mkdir()
        np.save(self.dataset / "vectors.npy", np.array([[1, 0], [0, 1]], dtype=np.float32))
        (self.dataset / "payloads.jsonl").write_text(
            "".join(json.dumps({"location": {"lat": lat, "lon": 0}}) + "\n" for lat in (0, 10))
        )
        (self.dataset / "tests.jsonl").write_text(
            '{"query":[1,0],"closest_ids":[0],"conditions":{"and":['
            '{"location":{"geo":{"lat":0,"lon":0,"radius":10}}}]}}\n'
        )
        self.profile = load_profile("milvus-hnsw-dense")
        self.settings = {
            "repo_path": str(self.source),
            "python_executable": sys.executable,
            "dataset_path": str(self.dataset),
            "dataset_entry": {"schema": {"location": "geo"}},
            "milvus_geo_filter": MILVUS_GEO_CONTRACT,
            "max_geo_filter_bytes": 4096,
        }
        self.execution = dict(host="localhost", filtered=True, distance="cosine", vector_size=2)
        candidate = {"hnsw.m": 32}
        rendered = ExperimentRenderer(self.profile).render(
            candidate, {"connection_params": {"port": 19531}}
        )
        workload = WorkloadSpec("local-geo", "cosine", 1, 1, 2, filtered=True)
        self.request = EvaluationRequest(
            "synthetic-geo", "milvus", candidate, rendered, workload, 42, 60
        )

    def runner(self, settings=None, execution=None):
        return VectorDBBenchmarkRunner(
            RunnerContext(
                self.profile,
                {**self.settings, **(settings or {})},
                {**self.execution, **(execution or {})},
                self.root / "artifacts",
            )
        )

    def test_evaluate_stages_geo_wrappers_dataset_and_auditable_parameters(self):
        runner = self.runner()
        rendered_before = json.dumps(self.request.rendered_experiment, sort_keys=True)

        def synthetic_process(argv, *, cwd, **kwargs):
            experiment_name = argv[argv.index("--engines") + 1]
            experiment = json.loads(
                (cwd / "experiments/configurations" / f"{experiment_name}.json").read_text()
            )[0]
            expected = json.loads(rendered_before)
            expected["name"] = experiment_name
            expected["search_params"][0]["eigen_geo"] = {
                "dataset_path": str(cwd / "datasets/local-data"),
                "schema": {"location": "geo"},
                "max_filter_bytes": 4096,
            }
            self.assertEqual(experiment, expected)
            registry = json.loads((cwd / "datasets/datasets.json").read_text())[0]
            self.assertEqual(
                registry,
                {
                    "name": "local-geo",
                    "distance": "cosine",
                    "vector_size": 2,
                    "type": "tar",
                    "path": "local-data",
                    "schema": {"location": "geo"},
                },
            )
            for name in ("vectors.npy", "payloads.jsonl", "tests.jsonl"):
                self.assertEqual(
                    (cwd / "datasets/local-data" / name).read_bytes(),
                    (self.dataset / name).read_bytes(),
                )
            directory = cwd / "engine/clients/milvus"
            self.assertIn("GeoRadiusFilter", (directory / "search.py").read_text())
            self.assertIn("local.config.schema = {}", (directory / "configure.py").read_text())
            self.assertIn("class MilvusSearcher", (directory / "eigen_search_base.py").read_text())
            self.assertIn(
                "class MilvusConfigurator", (directory / "eigen_configure_base.py").read_text()
            )
            self.assertTrue((directory / "eigen_geo.py").is_file())
            self.assertIn("preserve ANN index construction", (directory / "upload.py").read_text())
            self.assertNotIn("--skip-upload", argv)
            params = {"experiment": experiment_name, "engine": "milvus", "dataset": "local-geo"}
            for stage, metrics in (
                ("search", {"rps": 100, "mean_precisions": 1}),
                ("upload", {"total_time": 1}),
            ):
                (cwd / "results" / f"{experiment_name}-local-geo-{stage}-0-test.json").write_text(
                    json.dumps({"params": params, "results": metrics})
                )
            return 0, False, 0.001

        with patch(
            "eigen.runners.vectordb_benchmark._run_process", side_effect=synthetic_process
        ) as process:
            result = runner.evaluate(self.request)
        self.assertTrue(result.ok, result.error)
        process.assert_called_once()
        self.assertEqual(result.auxiliary["runner"]["milvus_geo_filter"], MILVUS_GEO_CONTRACT)
        self.assertEqual(result.auxiliary["runner"]["max_geo_filter_bytes"], 4096)
        command = next(Path(p) for p in result.artifacts if Path(p).name == "command.json")
        self.assertIn(
            "engine/clients/milvus/eigen_geo.py",
            json.loads(command.read_text())["compatibility_overlays"],
        )
        self.assertEqual(
            json.dumps(self.request.rendered_experiment, sort_keys=True), rendered_before
        )
        self.assertEqual(source_snapshot(self.source), self.original_source)

    def test_invalid_geo_contract_or_parameters_fail_before_evaluation(self):
        cases = [
            ({"milvus_geo_filter": "unknown-v2"}, {}, "unsupported"),
            ({}, {"filtered": False}, "filtered Milvus"),
            (
                {"dataset_entry": {"schema": {"location": "geo", "category": "keyword"}}},
                {},
                "exclusively geo",
            ),
        ]
        cases.extend(
            ({"max_geo_filter_bytes": value}, {}, "positive integer")
            for value in (0, -1, True, "4096")
        )
        for settings, execution, message in cases:
            with (
                self.subTest(settings=settings, execution=execution),
                self.assertRaisesRegex(ConfigurationError, message),
            ):
                self.runner(settings, execution)

    def test_unfiltered_request_cannot_reuse_a_geo_runner(self):
        request = replace(self.request, workload=replace(self.request.workload, filtered=False))
        with patch("eigen.runners.vectordb_benchmark._run_process") as process:
            result = self.runner().evaluate(request)
        self.assertFalse(result.ok)
        self.assertIn("different workload", result.error)
        process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
