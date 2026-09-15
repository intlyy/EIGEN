"""Offline regression cases for paper invariants; no database or API access."""

from __future__ import annotations

import json
import random
import tempfile
import threading
import unittest
from pathlib import Path

from mutune.api import BaseRunner, Observation, RunnerContext, RunStatus
from mutune.config import (
    ExecutionConfig,
    LoadedProject,
    MetricConstraint,
    ObjectiveSpec,
    ProjectConfig,
    TuningConfig,
)
from mutune.llm import CompletionResult, LLMError
from mutune.profiles import load_profile
from mutune.search_space import SearchSpace
from mutune.study import LoadedStudy, StudyConfig, rank_cross_validation, run_study
from mutune.tuning.history import EvaluationRecord, candidate_key
from mutune.tuning.llm_surrogate import LLMSurrogate
from mutune.tuning.pareto import archive, hypervolume
from mutune.tuning.partitioning import ProfilePartitioner
from mutune.tuning.tuner import Tuner


def record(i, qps, build, recall=0.95):
    return EvaluationRecord.from_observation(
        sequence=i,
        run_id=str(i),
        candidate={"x": i},
        observation=Observation(
            RunStatus.OK, {"qps": qps, "build_total_time_s": build, "recall": recall}
        ),
    )


def measured(candidate, qps, recall=0.95):
    return {"candidate": candidate, "status": "ok", "metrics": {"qps": qps, "recall": recall}}


class ParetoAndTransferTests(unittest.TestCase):
    def test_default_archive_maximizes_feasible_qps_without_build_time_tradeoff(self):
        config = TuningConfig()
        records = [
            record(0, 100, 1),
            record(1, 200, 1000),  # Higher feasible QPS wins despite slower construction.
            record(2, 999, 0.1, 0.8),  # Higher QPS cannot compensate for infeasibility.
            record(3, 200, 5),  # Build time does not break a QPS tie.
            record(4, 200, 10),
        ]
        del records[4].metrics["build_total_time_s"]  # This diagnostic is optional.
        front = archive(records, config.objectives, config.all_constraints())
        self.assertEqual([r.sequence for r in front], [1, 3, 4])

    def test_explicit_multiobjective_archive_respects_directions_and_recall(self):
        objectives = [
            ObjectiveSpec(),
            ObjectiveSpec(metric="build_total_time_s", direction="minimize"),
        ]
        records = [
            record(0, 100, 10),
            record(1, 200, 20),
            record(2, 90, 30),
            record(3, 999, 1, 0.1),
        ]
        front = archive(records, objectives, [MetricConstraint(metric="recall", threshold=0.9)])
        self.assertEqual([r.sequence for r in front], [0, 1])

    def test_hypervolume_is_union_not_sum(self):
        self.assertAlmostEqual(hypervolume([(1.0, 0.5), (0.5, 1.0)]), 0.75)
        self.assertAlmostEqual(hypervolume([(1.0, 1.0, 1.0), (0.5, 0.5, 0.5)]), 1.0)

    def test_any_view_infeasible_is_removed_before_normalization(self):
        candidates = [{"x": i} for i in range(3)]
        matrix = [
            [
                measured(candidates[0], 100),
                measured(candidates[1], 200),
                measured(candidates[2], 900),
            ],
            [
                measured(candidates[0], 100),
                measured(candidates[1], 300),
                measured(candidates[2], 1, 0.8),
            ],
        ]
        ranking = rank_cross_validation(
            candidates, matrix, constraints=[MetricConstraint(metric="recall", threshold=0.9)]
        )
        self.assertEqual(len(ranking), 2)
        self.assertEqual(ranking[0]["candidate"], candidates[1])
        self.assertEqual(ranking[0]["normalized_qps"], [1.0, 1.0])

    def test_stability_and_constant_dimensions_are_finite(self):
        candidates = [{"x": i} for i in range(3)]
        matrix = [
            [measured(c, q) for c, q in zip(candidates, values, strict=True)]
            for values in [[1, 11, 6], [11, 1, 6]]
        ]
        result = rank_cross_validation(
            candidates, matrix, constraints=[MetricConstraint(metric="recall", threshold=0.9)]
        )
        self.assertEqual(result[0]["candidate"], candidates[2])
        self.assertEqual(result[0]["performance_score"], 0)
        self.assertEqual(result[0]["stability_score"], 1)

    def test_missing_view_cell_is_rejected(self):
        with self.assertRaises(ValueError):
            rank_cross_validation([{"x": 1}], [[]], constraints=[])

    def test_parallel_tuning_all_view_validation_then_full_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = load_profile("qdrant-hnsw-dense")
            projects = []
            for i in range(3):
                cfg = ProjectConfig(
                    experiment_name=f"mini-{i}",
                    engine_profile=profile.id,
                    execution=ExecutionConfig(dataset=str(i)),
                    runner={"plugin": "dry-run"},
                    tuning=TuningConfig(strategy="random"),
                )
                projects.append(LoadedProject(cfg, profile, root / f"{i}.json", root / str(i)))
            full = projects[-1]
            study = LoadedStudy(
                StudyConfig(minidbs=["0", "1"], full_database="2", minidb_manifest="manifest"),
                projects[:2],
                full,
                root / "study",
                {},
            )
            barrier = threading.Barrier(2)
            calls = []
            candidates = [{"x": 0}, {"x": 1}]

            def tune(project):
                barrier.wait(timeout=5)  # Fails if independent workers become sequential.
                i = int(project.config.execution.dataset)
                return {
                    "complete": True,
                    "transfer_candidates": [{"candidate": candidates[i]}],
                    "resumed_evaluations": 0,
                    "llm_wall_s": 0.0,
                }

            def evaluate(project, pool):
                calls.append((project.config.execution.dataset, list(pool)))
                full_stage = project.config.execution.dataset == "2"
                return [
                    measured(c, (200 - 50 * c["x"]) if full_stage else 100 + c["x"]) for c in pool
                ]

            result = run_study(study, tune_fn=tune, evaluate_fn=evaluate)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["best_candidate"], candidates[0])  # Full-DB measurement wins.
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(len(pool) == 2 for _, pool in calls))
            self.assertEqual(calls[-1][0], "2")


class FakeLLM:
    def __init__(self, *, extra_metric=False):
        self.roles = []
        self.extra_metric = extra_metric
        self.requested_metrics = []

    def complete(self, prompt, *, deadline=None):
        request = json.loads(prompt)
        if "candidates" not in request:
            self.roles.append("proposer")
            return CompletionResult("[{}]")
        self.roles.append("surrogate")
        self.requested_metrics.append(set(request["output_contract"]["predictions"][0]["metrics"]))
        values = []
        for candidate in request["candidates"]:
            values.append(
                {
                    "id": candidate["id"],
                    "probability_feasible": 0.95,
                    "metrics": {
                        "qps": {"mean": 999999, "stddev": 1},
                        "recall": {"mean": 0.95, "stddev": 0.01},
                    },
                }
            )
            if self.extra_metric:
                values[-1]["metrics"]["build_total_time_s"] = {"mean": 1, "stddev": 0.1}
        return CompletionResult(
            json.dumps({"predictions": values}), {"prompt_tokens": 1, "completion_tokens": 1}
        )


class MeasuredRunner(BaseRunner):
    def evaluate(self, request):
        m = request.candidate["hnsw.m"]
        return Observation(RunStatus.OK, {"qps": 100 + m, "recall": 0.95, "build_total_time_s": m})


class CALMTests(unittest.TestCase):
    def test_llm_is_used_for_both_roles_and_archive_contains_only_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = load_profile("qdrant-hnsw-dense")
            llm = FakeLLM()
            config = TuningConfig(
                budget=5,
                initial_samples=2,
                proposals_per_round=4,
                evaluations_per_round=2,
                proposal_exploration=0.01,
            )
            runner = MeasuredRunner(RunnerContext(profile, {}, {}, root))
            result = Tuner(
                profile=profile,
                tuning=config,
                execution=ExecutionConfig(dataset="fake", distance="l2"),
                runner=runner,
                artifact_dir=root,
                experiment_name="calm",
                llm_client=llm,
            ).run()
            self.assertTrue(result.complete)
            self.assertIn("surrogate", llm.roles)
            self.assertIn("proposer", llm.roles)
            self.assertTrue(all(names == {"qps", "recall"} for names in llm.requested_metrics))
            self.assertTrue(result.pareto_candidates)
            self.assertTrue(all(item["metrics"]["qps"] < 1000 for item in result.pareto_candidates))
            self.assertTrue((root / "llm/calls.jsonl").exists())

    def test_surrogate_rejects_unmatched_predictions_or_extra_metrics(self):
        space = SearchSpace(load_profile("qdrant-hnsw-dense"))
        surrogate = LLMSurrogate(
            space,
            FakeLLM(extra_metric=True),
            objectives=[ObjectiveSpec()],
            constraints=[MetricConstraint(metric="recall", threshold=0.9)],
            task={},
        )
        candidate = space.canonicalize({})
        with self.assertRaises(ValueError):
            surrogate._parse({"predictions": []}, {candidate_key(candidate): candidate})
        with self.assertRaises(LLMError):
            # FakeLLM returns an extra build metric, violating the exact metric schema.
            surrogate.predict_many([candidate])

    def test_every_region_remains_reachable(self):
        partitioner = ProfilePartitioner(SearchSpace(load_profile("milvus-native-dense")))
        rng = random.Random(42)
        selected = {
            partitioner.select_regions(
                [],
                1,
                rng=rng,
                exploration_probability=1,
                objective_metric="qps",
                constraint_metric="recall",
                threshold=0.9,
                exploration_weight=0.2,
            )[0].id
            for _ in range(200)
        }
        self.assertEqual(selected, {r.id for r in partitioner.regions})


if __name__ == "__main__":
    unittest.main()
