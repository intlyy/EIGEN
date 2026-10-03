from __future__ import annotations

import tempfile
import unittest
from importlib.metadata import EntryPoint, EntryPoints
from pathlib import Path
from unittest import mock

from eigen import __version__, plugins
from eigen.api import BaseRunner, EvaluationRequest, Observation, RunnerContext
from eigen.errors import PluginError


class _ThirdPartyRunner(BaseRunner):
    PLUGIN_ID = "third-party"

    def evaluate(self, request: EvaluationRequest) -> Observation:  # pragma: no cover
        raise NotImplementedError


class _FakeEntryPoint:
    value = "fake.package:Runner"
    dist = None

    def __init__(self, name: str = "third-party") -> None:
        self.name = name
        self.loads = 0

    def load(self):
        self.loads += 1
        return _ThirdPartyRunner


class PluginDiscoveryTests(unittest.TestCase):
    def test_installed_entry_points_are_selected_from_eigen_group(self) -> None:
        entry_points = EntryPoints(
            [
                EntryPoint(
                    name="third-party",
                    value=f"{__name__}:_ThirdPartyRunner",
                    group="eigen.runners",
                ),
                EntryPoint(
                    name="unrelated",
                    value="unavailable.module:Runner",
                    group="another_application.runners",
                ),
            ]
        )
        with mock.patch.object(plugins.metadata, "entry_points", return_value=entry_points):
            discovered = plugins.discover_runners(include_builtins=False)
        self.assertEqual(set(discovered), {"third-party"})
        self.assertIs(discovered["third-party"].load_class(), _ThirdPartyRunner)

    def test_builtin_discovery_is_lazy_and_has_stable_ids(self) -> None:
        with mock.patch.object(plugins, "_entry_points", return_value=[]):
            discovered = plugins.discover_runners()
        self.assertGreaterEqual(set(discovered), {"dry-run", "vector-db-benchmark"})
        self.assertTrue(discovered["dry-run"].source.startswith("builtin:"))

    def test_entry_point_is_loaded_only_when_selected(self) -> None:
        entry_point = _FakeEntryPoint()
        with mock.patch.object(plugins, "_entry_points", return_value=[entry_point]):
            discovered = plugins.discover_runners(include_builtins=False)
        self.assertEqual(entry_point.loads, 0)
        self.assertIs(discovered["third-party"].load_class(), _ThirdPartyRunner)
        self.assertEqual(entry_point.loads, 1)

    def test_duplicate_entry_point_cannot_shadow_builtin(self) -> None:
        entry_point = _FakeEntryPoint(name="dry-run")
        with mock.patch.object(plugins, "_entry_points", return_value=[entry_point]):
            with self.assertRaisesRegex(PluginError, "Duplicate runner plugin ID"):
                plugins.discover_runners()
        self.assertEqual(entry_point.loads, 0)

    def test_create_builtin_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = RunnerContext(
                profile=None,
                settings={"metrics": {"qps": 42.0, "recall": 0.9}},
                execution={},
                artifact_dir=Path(temporary),
            )
            with mock.patch.object(plugins, "_entry_points", return_value=[]):
                runner = plugins.create_runner("dry-run", context)
            self.assertEqual(runner.PLUGIN_ID, "dry-run")
            self.assertEqual(type(runner).__module__, "eigen.runners.dry_run")
            self.assertEqual(runner.manifest()["eigen_version"], __version__)

    def test_unknown_runner_lists_available_plugins(self) -> None:
        with mock.patch.object(plugins, "_entry_points", return_value=[]):
            with self.assertRaisesRegex(PluginError, "Unknown runner plugin"):
                plugins.load_runner_class("missing")


if __name__ == "__main__":
    unittest.main()
