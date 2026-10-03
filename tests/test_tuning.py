from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eigen.api import BaseRunner, Observation, RunnerContext, RunStatus
from eigen.config import ExecutionConfig, TuningConfig
from eigen.errors import RunnerError
from eigen.models import SearchSpaceSpec
from eigen.profiles import load_profile
from eigen.search_space import SearchSpace
from eigen.tuning import (
    ConstraintAwareAcquisition,
    EvaluationRecord,
    HistoryStore,
    MixedSpaceKnnSurrogate,
    ProfilePartitioner,
    Tuner,
    TuningError,
)
from eigen.utils import canonical_json, fingerprint


class FakeRunner(BaseRunner):
    PLUGIN_ID = "fake-test-runner"
    PLUGIN_VERSION = "1"

    def __init__(self, context: RunnerContext) -> None:
        super().__init__(context)
        self.requests = []
        self.closed = False

    def evaluate(self, request):
        self.requests.append(request)
        ef_search = float(request.candidate["hnsw.ef_search"])
        connections = float(request.candidate["hnsw.m"])
        qps = 12_000.0 / (1.0 + ef_search / 1_000.0 + connections / 200.0)
        recall = min(1.0, 0.78 + ef_search / 4_000.0 + connections / 1_000.0)
        return Observation(
            status=RunStatus.OK,
            metrics={"qps": qps, "recall": recall},
            auxiliary={"fake": True},
        )

    def close(self) -> None:
        self.closed = True


def make_runner(profile, artifact_dir: Path) -> FakeRunner:
    return FakeRunner(
        RunnerContext(
            profile=profile,
            settings={},
            execution={},
            artifact_dir=artifact_dir,
        )
    )


class EndToEndTunerTests(unittest.TestCase):
    def test_budget_artifacts_runtime_bindings_and_resume(self) -> None:
        profile = load_profile("qdrant-hnsw-dense")
        tuning = TuningConfig(
            budget=5,
            initial_samples=2,
            proposals_per_round=4,
            evaluations_per_round=2,
            regions_per_round=1,
            recall_threshold=0.80,
            strategy="random",
            seed=17,
            history_limit=20,
            resume=True,
        )
        execution = ExecutionConfig(
            dataset="fake-100-cosine",
            distance="cosine",
            vector_size=100,
            top_k=7,
            upload_parallel=3,
            search_parallel=2,
            batch_size=64,
            connection_params={"timeout": 4},
        )

        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            runner = make_runner(profile, artifact_dir)
            callback_requests = []
            tuner = Tuner(
                profile=profile,
                tuning=tuning,
                execution=execution,
                runner=runner,
                artifact_dir=artifact_dir,
                experiment_name="e2e-test",
                before_evaluation=callback_requests.append,
                evaluation_timeout_s=10,
            )
            result = tuner.run()

            self.assertTrue(result.complete)
            self.assertEqual(result.resumed_evaluations, 0)
            self.assertEqual(result.llm_wall_s, 0)
            self.assertEqual(result.transfer_candidates, result.pareto_candidates)
            self.assertEqual(result.completed_evaluations, 5)
            self.assertEqual(len(runner.requests), 5)
            self.assertEqual(len(callback_requests), 5)
            self.assertTrue(runner.closed)
            self.assertIsNotNone(result.best_candidate)
            self.assertEqual(
                result.best_metrics["qps"],
                max(item["metrics"]["qps"] for item in result.pareto_candidates),
            )
            self.assertEqual(
                len({canonical_json(dict(request.candidate)) for request in runner.requests}),
                5,
            )

            for request in runner.requests:
                rendered = request.rendered_experiment
                self.assertEqual(rendered["name"], request.run_id)
                self.assertEqual(rendered["connection_params"]["timeout"], 4)
                self.assertEqual(rendered["upload_params"]["parallel"], 3)
                self.assertEqual(rendered["upload_params"]["batch_size"], 64)
                self.assertEqual(rendered["search_params"][0]["parallel"], 2)
                self.assertEqual(rendered["search_params"][0]["top"], 7)

            history_lines = (
                (artifact_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(len(history_lines), 5)
            checkpoint = json.loads((artifact_dir / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["remaining_budget"], 0)
            self.assertTrue((artifact_dir / "run_manifest.json").exists())
            self.assertTrue(any((artifact_dir / "rounds").glob("round-*.json")))

            resumed_runner = make_runner(profile, artifact_dir)
            resumed = Tuner(
                profile=profile,
                tuning=tuning,
                execution=execution,
                runner=resumed_runner,
                artifact_dir=artifact_dir,
                experiment_name="e2e-test",
                evaluation_timeout_s=10,
            ).run()
            self.assertTrue(resumed.complete)
            self.assertEqual(resumed.resumed_evaluations, 5)
            self.assertEqual(resumed.llm_wall_s, 0)
            self.assertEqual(resumed_runner.requests, [])
            self.assertTrue(resumed_runner.closed)

            changed_runner = FakeRunner(
                RunnerContext(
                    profile=profile,
                    settings={"surface": "changed"},
                    execution={},
                    artifact_dir=artifact_dir,
                )
            )
            changed = Tuner(
                profile=profile,
                tuning=tuning,
                execution=execution,
                runner=changed_runner,
                artifact_dir=artifact_dir,
                experiment_name="e2e-test",
                evaluation_timeout_s=10,
            )
            with self.assertRaisesRegex(TuningError, "incompatible"):
                changed.run()
            self.assertTrue(changed_runner.closed)

            # A v1 QPS-only archive is not a valid continuation of the new policy.
            manifest_path = artifact_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["resume_contract"]["optimizer_contract_version"], 3)
            manifest["resume_contract"].pop("optimizer_contract_version")
            manifest["resume_contract"].pop("guidance_objectives")
            manifest["resume_contract_hash"] = fingerprint(manifest["resume_contract"], length=64)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            legacy_runner = make_runner(profile, artifact_dir)
            legacy = Tuner(
                profile=profile,
                tuning=tuning,
                execution=execution,
                runner=legacy_runner,
                artifact_dir=artifact_dir,
                experiment_name="e2e-test",
                evaluation_timeout_s=10,
            )
            with self.assertRaisesRegex(TuningError, "incompatible"):
                legacy.run()
            self.assertEqual(legacy_runner.requests, [])

    def test_control_plane_failure_aborts_without_consuming_budget(self) -> None:
        profile = load_profile("qdrant-hnsw-dense")
        tuning = TuningConfig(
            budget=1,
            initial_samples=1,
            proposals_per_round=1,
            evaluations_per_round=1,
            strategy="random",
        )
        execution = ExecutionConfig(
            dataset="fake-100-cosine",
            distance="cosine",
            top_k=10,
        )

        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            runner = make_runner(profile, artifact_dir)

            def fail_before_evaluation(_request) -> None:
                raise RunnerError("database never became query-ready")

            tuner = Tuner(
                profile=profile,
                tuning=tuning,
                execution=execution,
                runner=runner,
                artifact_dir=artifact_dir,
                experiment_name="control-plane-failure",
                before_evaluation=fail_before_evaluation,
            )
            with self.assertRaisesRegex(RunnerError, "query-ready"):
                tuner.run()

            self.assertEqual(runner.requests, [])
            self.assertTrue(runner.closed)
            history_path = artifact_dir / "history.jsonl"
            self.assertFalse(history_path.exists())
            checkpoint = json.loads((artifact_dir / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["completed_evaluations"], 0)
            self.assertEqual(checkpoint["remaining_budget"], 1)


class HistoryTests(unittest.TestCase):
    def test_recovers_from_truncated_final_jsonl_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            history = HistoryStore(path)
            record = EvaluationRecord.from_observation(
                sequence=0,
                run_id="run-0",
                candidate={"x": 1},
                observation=Observation(
                    status=RunStatus.OK,
                    metrics={"qps": 10.0, "recall": 0.9},
                ),
            )
            history.append(record)
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"sequence":1')

            recovered = HistoryStore(path)
            self.assertEqual(recovered.evaluation_count, 1)
            self.assertTrue(recovered.contains({"x": 1}))
            recovered.append(
                EvaluationRecord.from_observation(
                    sequence=1,
                    run_id="run-1",
                    candidate={"x": 2},
                    observation=Observation(
                        status=RunStatus.OK,
                        metrics={"qps": 11.0, "recall": 0.91},
                    ),
                )
            )
            reloaded = HistoryStore(path)
            self.assertEqual(reloaded.evaluation_count, 2)
            self.assertTrue(reloaded.contains({"x": 2}))


class ProfileDrivenPartitionTests(unittest.TestCase):
    def test_regions_are_derived_from_activation_predicates(self) -> None:
        spec = SearchSpaceSpec.model_validate(
            {
                "domain": "conditional-test",
                "parameters": {
                    "algorithm": {
                        "kind": "categorical",
                        "default": "hnsw",
                        "choices": ["hnsw", "ivf"],
                        "effect": "index",
                        "bindings": [{"pointer": "/algorithm"}],
                    },
                    "hnsw.m": {
                        "kind": "integer",
                        "default": 16,
                        "bounds": [4, 64],
                        "effect": "index",
                        "active_if": [{"parameter": "algorithm", "op": "eq", "value": "hnsw"}],
                        "bindings": [{"pointer": "/hnsw/m"}],
                    },
                    "ivf.nlist": {
                        "kind": "integer",
                        "default": 100,
                        "bounds": [10, 1000],
                        "effect": "index",
                        "active_if": [{"parameter": "algorithm", "op": "eq", "value": "ivf"}],
                        "bindings": [{"pointer": "/ivf/nlist"}],
                    },
                },
            }
        )
        regions = ProfilePartitioner(SearchSpace(spec)).regions
        self.assertEqual(
            {region.fixed["algorithm"] for region in regions},
            {"hnsw", "ivf"},
        )
        self.assertEqual(len(regions), 2)

    def test_boolean_and_nested_inactive_selectors_do_not_create_dead_regions(self) -> None:
        spec = SearchSpaceSpec.model_validate(
            {
                "domain": "nested-conditional-test",
                "parameters": {
                    "enabled": {
                        "kind": "boolean",
                        "default": False,
                        "effect": "index",
                        "bindings": [{"pointer": "/enabled"}],
                    },
                    "algorithm": {
                        "kind": "categorical",
                        "default": "a",
                        "choices": ["a", "b"],
                        "effect": "index",
                        "active_if": [{"parameter": "enabled", "op": "eq", "value": True}],
                        "bindings": [{"pointer": "/algorithm"}],
                    },
                    "algorithm.depth": {
                        "kind": "integer",
                        "default": 2,
                        "bounds": [1, 4],
                        "effect": "index",
                        "active_if": [{"parameter": "algorithm", "op": "eq", "value": "a"}],
                        "bindings": [{"pointer": "/depth"}],
                    },
                },
            }
        )
        regions = ProfilePartitioner(SearchSpace(spec)).regions
        self.assertEqual(len(regions), 3)
        self.assertIn({"enabled": False}, [region.fixed for region in regions])
        self.assertIn(
            {"enabled": True, "algorithm": "a"},
            [region.fixed for region in regions],
        )
        self.assertIn(
            {"enabled": True, "algorithm": "b"},
            [region.fixed for region in regions],
        )


class SurrogateAndAcquisitionTests(unittest.TestCase):
    def test_feasibility_probability_is_continuous(self) -> None:
        profile = load_profile("milvus-hnsw-dense")
        space = SearchSpace(profile)
        candidates = [
            space.canonicalize({"hnsw.m": 16, "hnsw.ef_construction": 128, "hnsw.ef_search": 100}),
            space.canonicalize({"hnsw.m": 16, "hnsw.ef_construction": 128, "hnsw.ef_search": 300}),
        ]
        records = [
            EvaluationRecord.from_observation(
                sequence=index,
                run_id=f"run-{index}",
                candidate=candidate,
                observation=Observation(
                    status=RunStatus.OK,
                    metrics={
                        "qps": 1000.0 - index * 100.0,
                        "recall": 0.8 + index * 0.2,
                    },
                ),
            )
            for index, candidate in enumerate(candidates)
        ]
        query = space.canonicalize(
            {"hnsw.m": 16, "hnsw.ef_construction": 128, "hnsw.ef_search": 200}
        )
        prediction = (
            MixedSpaceKnnSurrogate(
                space,
                objective_metric="qps",
                constraint_metric="recall",
                k=2,
            )
            .fit(records)
            .predict(query)
        )
        acquisition = ConstraintAwareAcquisition(
            objective_metric="qps",
            constraint_metric="recall",
            constraint_threshold=0.9,
        )
        probability = acquisition.probability_feasible(prediction)
        self.assertGreater(probability, 0.01)
        self.assertLess(probability, 0.99)
        self.assertGreater(prediction.constraint.stddev, 0.0)


if __name__ == "__main__":
    unittest.main()
