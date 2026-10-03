from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eigen.cli import main
from eigen.errors import RunnerError


def project_payload(artifact_dir: Path) -> dict:
    return {
        "schema_version": 1,
        "experiment_name": "cli-test",
        "engine_profile": "qdrant-hnsw-dense",
        "artifact_dir": str(artifact_dir),
        "execution": {
            "dataset": "random-100",
            "distance": "cosine",
            "top_k": 10,
        },
        "runner": {
            "plugin": "dry-run",
            "settings": {"deterministic_jitter": 0.1},
        },
        "lifecycle": {
            "mode": "external",
            "settings": {"endpoint": "localhost", "ready_check": {"kind": "none"}},
        },
        "tuning": {
            "budget": 3,
            "initial_samples": 1,
            "proposals_per_round": 2,
            "evaluations_per_round": 1,
            "strategy": "random",
            "seed": 7,
        },
    }


class CliTests(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_list_profiles_and_plugins(self) -> None:
        code, output, error = self.invoke("profiles", "list")
        self.assertEqual(code, 0, error)
        self.assertIn("milvus-hnsw-dense", output)
        self.assertIn("qdrant-hnsw-dense", output)
        self.assertIn("pgvector-hnsw-dense", output)

        code, output, error = self.invoke("plugins", "list")
        self.assertEqual(code, 0, error)
        self.assertIn("dry-run", output)
        self.assertIn("vector-db-benchmark", output)

    def test_validate_render_and_tune_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "project.json"
            artifacts = root / "artifacts"
            config_path.write_text(
                json.dumps(project_payload(artifacts)),
                encoding="utf-8",
            )

            code, output, error = self.invoke("validate", str(config_path))
            self.assertEqual(code, 0, error)
            self.assertTrue(json.loads(output)["valid"])

            code, output, error = self.invoke(
                "render",
                str(config_path),
                "--candidate",
                '{"hnsw.m":32,"hnsw.ef_construction":256,"hnsw.ef_search":512}',
            )
            self.assertEqual(code, 0, error)
            rendered = json.loads(output)
            self.assertEqual(rendered["collection_params"]["hnsw_config"]["m"], 32)

            code, output, error = self.invoke(
                "evaluate",
                str(config_path),
                "--candidate",
                '{"hnsw.m":32,"hnsw.ef_construction":256,"hnsw.ef_search":512}',
                "--repeat",
                "3",
                "--dry-run",
            )
            self.assertEqual(code, 0, error)
            evaluation = json.loads(output)
            self.assertEqual(evaluation["requested_repeats"], 3)
            self.assertEqual(evaluation["successful_repeats"], 3)
            self.assertEqual(len(evaluation["observations"]), 3)
            self.assertIn("qps", evaluation["mean_metrics"])

            candidate_set = root / "candidate-set.json"
            candidate_set.write_text(
                json.dumps(
                    [
                        {
                            "hnsw.m": 16,
                            "hnsw.ef_construction": 128,
                            "hnsw.ef_search": 128,
                        },
                        {
                            "hnsw.m": 32,
                            "hnsw.ef_construction": 256,
                            "hnsw.ef_search": 512,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            code, output, error = self.invoke(
                "evaluate",
                str(config_path),
                "--candidate-set",
                str(candidate_set),
                "--repeat",
                "2",
                "--dry-run",
            )
            self.assertEqual(code, 0, error)
            batch_evaluation = json.loads(output)
            self.assertEqual(batch_evaluation["requested_candidates"], 2)
            self.assertEqual(batch_evaluation["successful_evaluations"], 4)
            self.assertEqual(len(batch_evaluation["candidate_summaries"]), 2)
            self.assertEqual(len(batch_evaluation["observations"]), 4)

            code, output, error = self.invoke("tune", str(config_path))
            self.assertEqual(code, 0, error)
            result = json.loads(output)
            self.assertEqual(json.loads((artifacts / "result.json").read_text()), result)
            self.assertEqual(result["metrics_origin"], "synthetic")
            self.assertGreaterEqual(result["invocation_wall_s"], 0)
            self.assertTrue(result["complete"])
            self.assertEqual(result["completed_evaluations"], 3)
            self.assertIn("[EIGEN] preparing", error)
            self.assertIn("simulated evaluation", error)
            self.assertIn("[EIGEN] completed sequence=", error)
            self.assertTrue((artifacts / "history.jsonl").is_file())
            self.assertTrue((artifacts / "run_manifest.json").is_file())

    def test_failed_tune_invalidates_a_previous_successful_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            config = root / "project.json"
            config.write_text(json.dumps(project_payload(artifacts)), encoding="utf-8")
            (artifacts / "result.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "complete": True,
                        "best_candidate": {"x": 1},
                        "best_metrics": {"qps": 123},
                    }
                ),
                encoding="utf-8",
            )
            with patch("eigen.cli.create_runner", side_effect=RunnerError("unavailable")):
                code, _, _ = self.invoke("tune", str(config))
            self.assertEqual(code, 2)
            result = json.loads((artifacts / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["best_candidate"])
            self.assertIsNone(result["best_metrics"])
            self.assertNotIn("complete", result)

    def test_invalid_candidate_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "project.json"
            config_path.write_text(
                json.dumps(project_payload(root / "artifacts")),
                encoding="utf-8",
            )
            code, _output, error = self.invoke(
                "render",
                str(config_path),
                "--candidate",
                '{"unknown":1}',
            )
            self.assertEqual(code, 2)
            self.assertIn("unknown candidate parameters", error)


if __name__ == "__main__":
    unittest.main()
