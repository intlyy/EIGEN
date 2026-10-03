"""Offline matrix, missing-result and source-publication regressions."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import package_source
from scripts.package_source import ROOT_FILES, source_files
from scripts.prepare_reproduction import (
    PAPER_ALLOCATION_VERSION,
    PAPER_DATASETS,
    PAPER_RECALLS,
    main,
)
from scripts.summarize_reproduction import summarize_plan, write_summary
from scripts.summarize_study import summarize


class ReproductionTests(unittest.TestCase):
    def fixture(self, root: Path, names=("tiny5m",), uniform=False):
        benchmark = root / "benchmark"
        benchmark.mkdir()
        (benchmark / "run.py").write_text("# fixture, never executed\n", encoding="utf-8")
        source = root / "source.hdf5"
        source.write_bytes(b"fixture, not benchmark data")
        entries = []
        for index in range(3):
            path = root / f"mini-{index}.hdf5"
            path.write_bytes(b"fixture, not benchmark data")
            entries.append(
                {
                    "id": f"mini-{index}",
                    "path": str(path),
                    "sha256": "fixture-checksum",
                    "sample_seed": 100 + index,
                }
            )
        manifest = {
            "source": str(source),
            "source_sha256": "fixture-checksum",
            "source_size": 100,
            "target_size": 10,
            "dimension": 128,
            "metric": "l2",
            "top_k": 100,
            "method": "shared-p-stable-stratified",
            "minidbs": entries,
            "sampling_method": "locality-stratified",
            "allocation_version": PAPER_ALLOCATION_VERSION,
        }
        path = root / "stratified.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        variants = [{"label": "stratified-m3-r10", "path": str(path)}]
        if uniform:
            path = root / "uniform.json"
            path.write_text(
                json.dumps(
                    {
                        **manifest,
                        "method": "uniform-without-replacement",
                        "sampling_method": "uniform",
                    }
                ),
                encoding="utf-8",
            )
            variants.append({"label": "uniform-m3-r10", "path": str(path)})
        inventory = root / "inventory.json"
        inventory.write_text(
            json.dumps(
                {
                    "datasets": [
                        {"name": name, "description": "Offline fixture only", "manifests": variants}
                        for name in names
                    ]
                }
            ),
            encoding="utf-8",
        )
        return inventory, benchmark

    def generate(self, root, inventory, benchmark, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            main(
                [
                    "--inventory",
                    str(inventory),
                    "--benchmark-repo",
                    str(benchmark),
                    "--output",
                    str(root / "matrix"),
                    *extra,
                ]
            )
        return json.loads((root / "matrix/run-plan.json").read_text(encoding="utf-8"))

    def test_six_dataset_seven_recall_matrix_does_not_execute_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, benchmark = self.fixture(root, sorted(PAPER_DATASETS))
            plan = self.generate(root, inventory, benchmark, "--require-paper-datasets")
            self.assertEqual(len(plan["runs"]), 42)
            self.assertEqual({run["recall_threshold"] for run in plan["runs"]}, set(PAPER_RECALLS))
            self.assertEqual(plan["results_status"], "not_executed")
            self.assertFalse(any(Path(run["artifact_dir"]).exists() for run in plan["runs"]))
            run = plan["runs"][0]
            project = json.loads((Path(run["config"]).parent / "milvus-mini-00.json").read_text())
            self.assertIsNone(project["tuning"]["initial_samples"])
            self.assertEqual(project["tuning"]["recall_threshold"], run["recall_threshold"])
            self.assertEqual(
                project["runner"]["settings"]["expected_source_sha256"],
                plan["benchmark_source_sha256"],
            )

    def test_methods_use_real_manifest_types_and_explicit_direct_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, benchmark = self.fixture(root, uniform=True)
            plan = self.generate(
                root,
                inventory,
                benchmark,
                "--recalls",
                "0.95",
                "--methods",
                "eigen",
                "mean-only",
                "uniform-mini",
                "direct-full",
                "--direct-full-budget",
                "77",
            )
            self.assertEqual(len(plan["runs"]), 4)
            for run in plan["runs"]:
                config = json.loads(Path(run["config"]).read_text())
                if run["method"] == "mean-only":
                    self.assertEqual(config["stability_weight"], 0)
                elif run["method"] == "uniform-mini":
                    self.assertIn("uniform", run["sampling_method"])
                elif run["method"] == "direct-full":
                    self.assertEqual(config["tuning"]["budget"], 77)
                    self.assertEqual(run["num_minidbs"], 0)
                    self.assertIn("tune", run["run_argv"])

    def test_uniform_is_not_silently_substituted_and_direct_budget_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, benchmark = self.fixture(root)
            for method in ("uniform-mini", "direct-full"):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.generate(root, inventory, benchmark, "--methods", method)
                self.assertFalse((root / "matrix").exists())

    def test_missing_synthetic_failed_and_resumed_results_are_not_complete_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = [
                None,
                {
                    "status": "ok",
                    "metrics_origin": "synthetic",
                    "complete": True,
                    "resumed_evaluations": 0,
                    "invocation_wall_s": 4,
                    "best_metrics": {"qps": 300, "recall": 0.99},
                },
                {
                    "status": "failed",
                    "metrics_origin": "measured",
                    "complete": True,
                    "resumed_evaluations": 0,
                    "invocation_wall_s": 4,
                    "best_metrics": {"qps": 300, "recall": 0.99},
                },
                {
                    "status": "ok",
                    "metrics_origin": "measured",
                    "complete": True,
                    "resumed_evaluations": 3,
                    "invocation_wall_s": 4,
                    "best_metrics": {"qps": 300, "recall": 0.99},
                },
                {
                    "status": "ok",
                    "metrics_origin": "measured",
                    "complete": True,
                    "resumed_evaluations": 0,
                    "invocation_wall_s": 4,
                    "best_metrics": {"qps": 300, "recall": 0.99},
                },
                {
                    "complete": True,
                    "invocation_wall_s": 4,
                    "best_metrics": {"qps": 300, "recall": 0.99},
                },
            ]
            runs = []
            for index, result in enumerate(results):
                artifact = root / f"artifact-{index}"
                if result is not None:
                    artifact.mkdir()
                    (artifact / "result.json").write_text(json.dumps(result), encoding="utf-8")
                runs.append(
                    {
                        "id": str(index),
                        "method": "direct-full",
                        "artifact_dir": str(artifact),
                        "recall_threshold": 0.95,
                    }
                )
            plan = root / "plan.json"
            plan.write_text(json.dumps({"runs": runs}), encoding="utf-8")
            rows = summarize_plan(plan)
            self.assertEqual(rows[0]["status"], "missing")
            self.assertEqual(rows[1]["status"], "synthetic")
            for index in (0, 1, 2, 5):
                self.assertIsNone(rows[index]["qps"])
                self.assertIsNone(rows[index]["paper_aggregate_total_s"])
            self.assertIsNone(rows[3]["paper_aggregate_total_s"])
            self.assertEqual(rows[3]["qps"], 300)
            self.assertEqual(rows[4]["paper_aggregate_total_s"], 4)
            self.assertIsNone(rows[4]["paper_components_s"])
            write_summary(rows, root / "summary")
            self.assertIn('"qps": null', (root / "summary/results.json").read_text())
            self.assertTrue((root / "summary/results.csv").is_file())

    def test_old_stratified_allocation_cannot_be_labeled_current_paper_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, benchmark = self.fixture(root)
            path = root / "stratified.json"
            manifest = json.loads(path.read_text())
            del manifest["allocation_version"]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.generate(root, inventory, benchmark)
            self.assertFalse((root / "matrix").exists())

    def test_nonobject_artifacts_do_not_abort_matrix_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = [[], None, {"status": "ok", "best_metrics": []}]
            runs = []
            for index, result in enumerate(malformed):
                artifact = root / str(index)
                artifact.mkdir()
                (artifact / "result.json").write_text(json.dumps(result), encoding="utf-8")
                runs.append(
                    {
                        "id": str(index),
                        "method": "eigen",
                        "artifact_dir": str(artifact),
                        "recall_threshold": 0.95,
                    }
                )
            plan = root / "plan.json"
            plan.write_text(json.dumps({"runs": runs}), encoding="utf-8")
            rows = summarize_plan(plan)
            self.assertEqual([row["status"] for row in rows], ["invalid_artifact"] * 3)
            self.assertTrue(
                all(row["qps"] is None and row["total_cost_usd"] is None for row in rows)
            )

    def test_missing_model_logs_are_unknown_cost_and_zero_calls_require_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "status": "ok",
                "local_results": [{"resumed_evaluations": 0, "llm_wall_s": 10}],
            }
            (root / "result.json").write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(summarize(root)["total_cost_usd"])
            self.assertFalse(summarize(root)["llm_logs_complete"])
            payload["local_results"][0]["llm_wall_s"] = 0
            (root / "result.json").write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(summarize(root)["total_cost_usd"], 0)

    def test_partial_worker_model_logs_cannot_be_reported_as_complete_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "status": "ok",
                "local_results": [
                    {"resumed_evaluations": 0, "llm_wall_s": 10},
                    {"resumed_evaluations": 0, "llm_wall_s": 10},
                ],
            }
            (root / "result.json").write_text(json.dumps(payload), encoding="utf-8")
            for index, cost in enumerate((0.05, 0.10)):
                path = root / f"mini-{index:02d}/tuning/llm/calls.jsonl"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps({"usage": {"prompt_tokens": 1}, "cost_usd": cost}) + "\n",
                    encoding="utf-8",
                )
                summary = summarize(root)
                if index == 0:
                    self.assertIsNone(summary["total_cost_usd"])
                    self.assertEqual(summary["known_cost_subtotal_usd"], 0.05)
                else:
                    self.assertAlmostEqual(summary["total_cost_usd"], 0.15)

    def test_publication_archive_has_eigen_package_deployment_and_one_official_paper(self):
        repository = Path(__file__).resolve().parents[1]
        files = source_files(repository)
        self.assertEqual(
            [path.relative_to(repository).as_posix() for path in files if path.suffix == ".pdf"],
            ["docs/EIGEN_VLDB.pdf"],
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "EIGEN-paper-source.zip"
            with (
                patch("sys.argv", ["package_source.py", "--output", str(destination)]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                package_source.main()
            with zipfile.ZipFile(destination) as archive:
                names = set(archive.namelist())
                self.assertTrue(all(name.startswith("EIGEN/") for name in names))
                expected = {
                    "EIGEN/" + path.relative_to(repository).as_posix()
                    for path in (repository / "src/eigen").rglob("*.py")
                    if "__pycache__" not in path.parts
                }
                self.assertTrue(expected)
                self.assertTrue(expected.issubset(names))
                for name in (
                    "EIGEN/docs/EIGEN_VLDB.pdf",
                    "EIGEN/src/eigen/resources/profiles/milvus-native-dense.json",
                    "EIGEN/src/eigen/resources/vectordb_benchmark/milvus_geo_search.py.txt",
                    "EIGEN/examples/paper/deploy/pgvector.Dockerfile",
                ):
                    self.assertIn(name, names)
                self.assertTrue(archive.read("EIGEN/docs/EIGEN_VLDB.pdf").startswith(b"%PDF-"))

    def test_source_selection_excludes_environment_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ROOT_FILES:
                (root / name).write_text("fixture", encoding="utf-8")
            (root / "scripts").mkdir()
            (root / "scripts/.env.json").write_text("not-a-real-secret", encoding="utf-8")
            self.assertNotIn(root / "scripts/.env.json", source_files(root))


if __name__ == "__main__":
    unittest.main()
