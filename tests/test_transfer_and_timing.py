"""Regressions for complete frontier transfer and non-overlapping cost accounting."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mutune.api import Observation, RunStatus
from mutune.config import TuningConfig
from mutune.llm import CompletionResult, LoggedCompletionClient
from mutune.study import rank_cross_validation
from mutune.timing import ConstructionTimer, finalize_study_timings
from mutune.tuning.history import EvaluationRecord, candidate_key
from mutune.tuning.pareto import archive
from mutune.tuning.transfer import transfer_pool


def record(name, qps, recall, region="hnsw"):
    return EvaluationRecord.from_observation(
        sequence=ord(name),
        run_id=name,
        candidate={"name": name},
        region_id=region,
        observation=Observation(RunStatus.OK, {"qps": qps, "recall": recall}),
    )


class TransferTests(unittest.TestCase):
    def test_safe_slightly_slower_backup_survives_local_pruning_and_cross_validation(self):
        config = TuningConfig()
        records = [
            [record("A", 100, 0.95), record("B", 99, 0.99)],
            [record("C", 100, 0.95), record("B", 99, 0.99)],
        ]
        fronts = [
            archive(rows, config.guidance_objectives(), config.all_constraints())
            for rows in records
        ]
        self.assertEqual(
            [r.candidate["name"] for rows in fronts for r in rows], ["A", "B", "C", "B"]
        )
        union = {
            candidate_key(r.candidate): r.candidate
            for rows in records
            for r in transfer_pool(rows, config)
        }
        candidates = list(union.values())
        self.assertEqual({c["name"] for c in candidates}, {"A", "B", "C"})
        matrix = [
            [
                {
                    "candidate": c,
                    "status": "ok",
                    "metrics": {
                        "qps": 99 if c["name"] == "B" else 100,
                        "recall": 0.99
                        if c["name"] == "B"
                        else 0.95
                        if c["name"] == local
                        else 0.89,
                    },
                }
                for c in candidates
            ]
            for local in ("A", "C")
        ]
        ranking = rank_cross_validation(candidates, matrix, constraints=config.all_constraints())
        self.assertEqual([r["candidate"]["name"] for r in ranking], ["B"])

    def test_pool_keeps_entire_frontier_despite_legacy_region_cap(self):
        config = TuningConfig(transfer_candidates_per_region=2)
        rows = [
            record("A", 100, 0.95),
            record("B", 99, 0.96),
            record("C", 98, 0.98),
            record("D", 999, 0.89),
            record("E", 97, 1),
            record("F", 4, 0.99, "ivf"),  # Dominated points do not get region reservations.
        ]
        selected = transfer_pool(rows, config)
        self.assertEqual({r.candidate["name"] for r in selected}, {"A", "B", "C", "E"})
        self.assertEqual(
            selected, archive(rows, config.guidance_objectives(), config.all_constraints())
        )
        self.assertEqual([o.metric for o in config.objectives], ["qps"])


class TimingTests(unittest.TestCase):
    def test_construction_includes_sequential_stages_without_overlap(self):
        with patch("mutune.timing.time.perf_counter", side_effect=[10, 12, 17, 20]):
            timer = ConstructionTimer()
            timer.mark("input")
            timer.mark("ground_truth")
            result = timer.finish()
        self.assertEqual(result["wall_s"], 10)
        self.assertEqual(sum(result["stages_wall_s"].values()), 10)

    def test_parallel_critical_path_decomposition_does_not_sum_worker_llm_times(self):
        timings = {
            "parallel_tuning_wall_s": 10,
            "critical_path_worker": 1,
            "cross_validation_wall_s": 3,
            "full_validation_wall_s": 5,
        }
        results = [
            {"resumed_evaluations": 0, "llm_wall_s": 8},
            {"resumed_evaluations": 0, "llm_wall_s": 2},
        ]
        finalize_study_timings(timings, {"construction_timing": {"wall_s": 20}}, results, 19)
        self.assertEqual(timings["cold_start_pipeline_wall_s"], 39)
        self.assertEqual(sum(timings["additive_wall_s"].values()), 39)
        self.assertEqual(timings["llm_worker_sum_s"], 10)
        self.assertEqual(timings["llm_critical_path_wall_s"], 2)
        self.assertEqual(timings["additive_wall_s"]["minidb_tuning_without_critical_path_llm"], 8)

    def test_missing_construction_or_resumed_history_never_claims_cold_start_total(self):
        for manifest, resumed in (({}, 0), ({"construction_timing": {"wall_s": 20}}, 3)):
            timings = {"parallel_tuning_wall_s": 10, "critical_path_worker": 0}
            finalize_study_timings(
                timings, manifest, [{"resumed_evaluations": resumed, "llm_wall_s": 1}], 12
            )
            self.assertEqual(timings["total_invocation_wall_s"], 12)
            self.assertIsNone(timings["cold_start_pipeline_wall_s"])
            self.assertIsNone(timings["additive_wall_s"])

    def test_llm_accumulator_counts_failures_but_not_previous_process_logs(self):
        class Client:
            def complete(self, *args, **kwargs):
                return CompletionResult(content="{}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calls.jsonl").write_text(json.dumps({"elapsed_s": 999}) + "\n")
            client = LoggedCompletionClient(Client(), root, "test")
            with patch("mutune.llm.time.monotonic", side_effect=[0, 2]):
                client.complete("{}")
            with patch.object(client.client, "complete", side_effect=ValueError("failed")):
                with patch("mutune.llm.time.monotonic", side_effect=[3, 6]):
                    with self.assertRaises(ValueError):
                        client.complete("{}")
            self.assertEqual(client.elapsed_s, 5)


if __name__ == "__main__":
    unittest.main()
