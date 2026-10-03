"""Offline boundary regressions; benchmark process outputs are synthetic."""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from eigen.api import EvaluationRequest, RunnerContext, WorkloadSpec
from eigen.benchmark_compat import install_benchmark_compatibility
from eigen.errors import ConfigurationError, RunnerError
from eigen.profiles import load_profile
from eigen.rendering import ExperimentRenderer
from eigen.runners.vectordb_benchmark import VectorDBBenchmarkRunner

BASE = """import time
DEFAULT_TOP = 10
class BaseSearcher:
    @classmethod
    def _search_one(cls, query, top=None):
        return -1, 0  # An old result calculation to replace.
    def unrelated(self):
        return "preserved"
"""
CONNECT = """def connect(host, connection_params):
    return connections.connect(alias=MILVUS_DEFAULT_ALIAS, host=host,
        port=str(connection_params.get("port", MILVUS_DEFAULT_PORT)), **connection_params)
"""


def benchmark_fixture(root):
    for name in (
        "benchmark",
        "engine/base_client",
        "dataset_reader",
        "datasets",
        "engine/clients/milvus",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "datasets/datasets.json").write_text("[]")
    (root / "run.py").write_text("# Synthetic CLI outputs are supplied by the test.\n")
    (root / "engine/base_client/search.py").write_text(BASE)
    for name in ("configure", "upload", "search"):
        (root / f"engine/clients/milvus/{name}.py").write_text(CONNECT)


class CompatibilityTests(unittest.TestCase):
    def test_milvus_all_connection_phases_accept_port_and_preserve_params(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_fixture(root)
            changed = install_benchmark_compatibility(root, "milvus")
            self.assertEqual(len(changed), 4)
            for name in ("configure", "upload", "search"):
                namespace = {
                    "connections": types.SimpleNamespace(connect=lambda **kw: kw),
                    "MILVUS_DEFAULT_ALIAS": "default",
                    "MILVUS_DEFAULT_PORT": 19530,
                }
                exec((root / f"engine/clients/milvus/{name}.py").read_text(), namespace)
                for params in ({"port": 19531, "user": "test"}, {"user": "test"}):
                    original = params.copy()
                    result = namespace["connect"]("localhost", params)
                    self.assertEqual(result["port"], str(params.get("port", 19530)))
                    self.assertEqual(result["user"], "test")
                    self.assertEqual(params, original)
            # Reapplying an adapter update must not add a second keyword.
            install_benchmark_compatibility(root, "milvus")

    def test_recall_uses_available_exact_answers_and_validates_empty_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_fixture(root)
            install_benchmark_compatibility(root, "qdrant")
            namespace = {}
            exec((root / "engine/base_client/search.py").read_text(), namespace)
            cls = namespace["BaseSearcher"]
            self.assertEqual(cls().unrelated(), "preserved")
            for expected, returned, score in (
                ([1, 2, 3], [1, 2, 3], 1.0),
                ([1, 2, 3], [1, 1, 2], 2 / 3),
                ([], [999], 0.0),
                ([], [], 1.0),
                (list(range(10)), [0, 1, 2], 0.3),
            ):
                with self.subTest(expected=expected, returned=returned):
                    cls.search_one = lambda query, top, ids=returned: [(i, 0) for i in ids]
                    query = types.SimpleNamespace(
                        expected_result=expected, meta_conditions={"geo": {}}
                    )
                    recall, elapsed = cls._search_one(query, 10)
                    self.assertEqual(recall, score)
                    self.assertGreaterEqual(elapsed, 0)
            for expected, filtered in ((None, True), ([1], False), ([], False), ([1, 1], True)):
                query = types.SimpleNamespace(
                    expected_result=expected, meta_conditions={"geo": {}} if filtered else None
                )
                with self.assertRaises(ValueError):
                    cls._search_one(query, 10)

    def test_unknown_base_search_signature_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_fixture(root)
            path = root / "engine/base_client/search.py"
            path.write_text(BASE.replace("top=None", "limit=None"))
            with self.assertRaisesRegex(RunnerError, "signature"):
                install_benchmark_compatibility(root, "qdrant")
            ast.parse(path.read_text())


class FreshMeasurementTests(unittest.TestCase):
    def test_manual_skip_flags_are_rejected_even_without_state_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_fixture(root / "benchmark")
            for name in ("skip_upload", "skip_configure", "skip_search"):
                for reuse in (False, True):
                    with self.subTest(flag=name, reuse=reuse):
                        with self.assertRaisesRegex(ConfigurationError, "Manual skip"):
                            VectorDBBenchmarkRunner(
                                RunnerContext(
                                    load_profile("qdrant-hnsw-dense"),
                                    {
                                        "repo_path": str(root / "benchmark"),
                                        name: True,
                                        "state_reuse": reuse,
                                    },
                                    {},
                                    root / "artifacts",
                                )
                            )

    def test_fresh_evaluation_requires_identified_build_evidence(self):
        for upload, expected_ok in (
            (None, False),
            ({}, False),
            ({"total_time": -1}, False),
            ({"total_time": 2.5}, True),
        ):
            with self.subTest(upload=upload), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "benchmark"
                benchmark_fixture(source)
                profile = load_profile("qdrant-hnsw-dense")
                runner = VectorDBBenchmarkRunner(
                    RunnerContext(
                        profile,
                        {
                            "repo_path": str(source),
                            "python_executable": sys.executable,
                            "keep_workspace": False,
                        },
                        {"host": "localhost"},
                        root / "artifacts",
                    )
                )
                runner._materialize_dataset = lambda *args: None

                def process(argv, *, cwd, upload_metrics=upload, **kwargs):
                    self.assertNotIn("--skip-upload", argv)
                    experiment = argv[argv.index("--engines") + 1]
                    dataset = argv[argv.index("--datasets") + 1]
                    params = {"experiment": experiment, "engine": "qdrant", "dataset": dataset}
                    for stage, metrics in (
                        ("search", {"rps": 100, "mean_precisions": 0.95}),
                        ("upload", upload_metrics),
                    ):
                        if metrics is not None:
                            (
                                cwd / "results" / f"{experiment}-{dataset}-{stage}-0-test.json"
                            ).write_text(json.dumps({"params": params, "results": metrics}))
                    return 0, False, 0.001

                request = EvaluationRequest(
                    "test",
                    "qdrant",
                    {"hnsw.m": 32},
                    ExperimentRenderer(profile).render({"hnsw.m": 32}),
                    WorkloadSpec("fixture", "l2", 10, 1, 4),
                    42,
                    60.0,
                )
                with patch("eigen.runners.vectordb_benchmark._run_process", side_effect=process):
                    result = runner.evaluate(request)
                self.assertEqual(result.ok, expected_ok, result.error)
                if expected_ok:
                    self.assertEqual(result.metrics["build_total_time_s"], 2.5)
                    self.assertFalse(result.auxiliary["data_reused"])
                # Compatibility installation must never alter the user's checkout.
                self.assertEqual((source / "engine/base_client/search.py").read_text(), BASE)


if __name__ == "__main__":
    unittest.main()
