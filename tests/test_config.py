from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from eigen.config import ProjectConfig, load_project
from eigen.errors import ConfigurationError


def valid_payload() -> dict:
    return {
        "schema_version": 1,
        "experiment_name": "unit-test",
        "engine_profile": "qdrant-hnsw-dense",
        "artifact_dir": "./artifacts",
        "execution": {
            "dataset": "random-100",
            "distance": "cosine",
            "top_k": 10,
        },
        "runner": {"plugin": "dry-run", "settings": {}},
        "tuning": {
            "budget": 5,
            "initial_samples": 2,
            "proposals_per_round": 4,
            "evaluations_per_round": 2,
            "strategy": "random",
        },
    }


class ProjectConfigTests(unittest.TestCase):
    def test_strict_unknown_field(self) -> None:
        payload = valid_payload()
        payload["unknown"] = True
        with self.assertRaises(ValidationError):
            ProjectConfig.model_validate(payload)

    def test_llm_strategy_requires_llm_section(self) -> None:
        payload = valid_payload()
        payload["tuning"]["strategy"] = "llm"
        with self.assertRaises(ValidationError):
            ProjectConfig.model_validate(payload)

    def test_plaintext_api_key_is_not_a_supported_field(self) -> None:
        payload = valid_payload()
        payload["llm"] = {
            "model": "example",
            "api_key": "must-not-be-stored-here",
        }
        with self.assertRaises(ValidationError):
            ProjectConfig.model_validate(payload)

    def test_llm_extra_body_cannot_override_core_request(self) -> None:
        payload = valid_payload()
        payload["llm"] = {
            "model": "example",
            "extra_body": {"messages": [{"role": "user", "content": "replace"}]},
        }
        with self.assertRaises(ValidationError):
            ProjectConfig.model_validate(payload)

    def test_llm_sampling_fields_may_be_omitted_from_wire_payload(self) -> None:
        payload = valid_payload()
        payload["llm"] = {
            "model": "thinking-model",
            "temperature": None,
            "max_tokens": None,
        }
        config = ProjectConfig.model_validate(payload)
        self.assertIsNone(config.llm.temperature)
        self.assertIsNone(config.llm.max_tokens)

    def test_relative_paths_resolve_from_project_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "configs" / "project.json"
            project_path.parent.mkdir()
            payload = valid_payload()
            payload["artifact_dir"] = "../run-artifacts"
            payload["runner"]["settings"]["repo_path"] = "../../vector-db-benchmark"
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_project(project_path)
            self.assertEqual(loaded.artifact_dir, (root / "run-artifacts").resolve())
            self.assertEqual(
                loaded.config.runner.settings["repo_path"],
                str((root.parent / "vector-db-benchmark").resolve()),
            )

    def test_invalid_json_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_project(path)

    def test_bare_python_executable_remains_a_path_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "project.json"
            payload = valid_payload()
            payload["runner"]["settings"]["python_executable"] = "python3"
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_project(project_path)
            self.assertEqual(
                loaded.config.runner.settings["python_executable"],
                "python3",
            )

    def test_relative_workspace_remains_artifact_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_path = Path(temporary) / "project.json"
            payload = valid_payload()
            payload["runner"]["settings"]["workspace_dir"] = "workspace"
            project_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_project(project_path)
            self.assertEqual(loaded.config.runner.settings["workspace_dir"], "workspace")


if __name__ == "__main__":
    unittest.main()
