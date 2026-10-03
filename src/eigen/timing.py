"""Paper aggregate tuning cost and separate elapsed wall-clock diagnostics."""

from __future__ import annotations

import math
import time


class ConstructionTimer:
    """Disjoint, sequential stages, including reads, writes and checksums."""

    def __init__(self):
        self.started = self.previous = time.perf_counter()
        self.stages = {}

    def mark(self, name):
        now = time.perf_counter()
        self.stages[name] = self.stages.get(name, 0.0) + now - self.previous
        self.previous = now

    def finish(self):
        self.mark("manifest_preparation")
        return {
            "wall_s": self.previous - self.started,
            "stages_wall_s": dict(self.stages),
            "scope": "all views, input reads through checksums; excludes final manifest write",
        }


def _seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def finalize_study_timings(timings, manifest, local_results, invocation_wall_s):
    """Charge each concurrent worker separately (paper Sections 6.1 and 6.7).

    Wall time remains useful operationally, but does not replace the paper's
    aggregate optimization cost. A resumed invocation or old uninstrumented
    artifact cannot establish the cost of a complete fresh tuning run.
    """
    construction = manifest.get("construction_timing", {}).get("wall_s")
    construction = construction if _seconds(construction) else None
    resumed = [result.get("resumed_evaluations") for result in local_results]
    cold_start = all(type(count) is int and count == 0 for count in resumed)
    llm = [result.get("llm_wall_s") for result in local_results]
    llm_known = all(_seconds(value) for value in llm)
    critical = timings["critical_path_worker"]
    parallel = timings["parallel_tuning_wall_s"]
    llm_critical = llm[critical] if llm_known else None
    if llm_critical is not None and llm_critical > parallel:
        raise ValueError("worker LLM time cannot exceed the parallel tuning stage")
    total = construction + invocation_wall_s if cold_start and construction is not None else None
    full = timings.get("full_validation_wall_s", 0.0)
    local_workers = timings.get("local_optimization_worker_wall_s")
    cross_workers = timings.get("cross_validation_worker_wall_s")
    aggregation = timings.get("aggregation_wall_s")
    worker_times_known = all(
        isinstance(values, list)
        and len(values) == len(local_results)
        and all(_seconds(value) for value in values)
        for values in (local_workers, cross_workers)
    )
    local_sum = sum(local_workers) if worker_times_known else None
    cross_sum = sum(cross_workers) if worker_times_known else None
    llm_sum = sum(llm) if llm_known else None
    if (
        worker_times_known
        and llm_known
        and any(
            inference > worker + 1e-6 for inference, worker in zip(llm, local_workers, strict=True)
        )
    ):
        raise ValueError("worker LLM time cannot exceed that worker's optimization time")
    paper_components = None
    if (
        cold_start
        and construction is not None
        and worker_times_known
        and llm_known
        and _seconds(aggregation)
        and _seconds(full)
    ):
        paper_components = {
            "minidb_construction": construction,
            "minidb_tuning": max(0.0, local_sum - llm_sum) + cross_sum,
            "calm_inference": llm_sum,
            "cross_minidb_aggregation": aggregation,
            "full_database_validation": full,
        }
    timings.update(
        total_invocation_wall_s=invocation_wall_s,
        construction_wall_s=construction,
        resumed_evaluations=resumed,
        cold_start=cold_start,
        cold_start_pipeline_wall_s=total,
        llm_worker_sum_s=sum(llm) if llm_known else None,
        llm_critical_path_wall_s=llm_critical,
        local_optimization_worker_sum_s=local_sum,
        cross_validation_worker_sum_s=cross_sum,
        paper_components_s=paper_components,
        paper_aggregate_total_s=(sum(paper_components.values()) if paper_components else None),
        paper_accounting=(
            "Sections 6.1/6.7: construction + sum of local worker optimization times "
            "(including LLM calls) + sum of physical cross-Mini-DB worker times + "
            "numerical candidate deduplication/ranking + full-database validation. "
            "MiniDB tuning includes non-LLM local optimization work and physical "
            "cross-Mini-DB measurements; CALM inference sums all worker LLM time. "
            "Orchestrator I/O and scheduling overhead remain wall-time diagnostics. "
            "A complete paper cost is unavailable for resumed or uninstrumented runs."
        ),
        additive_wall_s=(
            {
                "minidb_construction": construction,
                "minidb_tuning_without_critical_path_llm": parallel - llm_critical,
                "llm_on_critical_path": llm_critical,
                "cross_validation_aggregation_and_control": invocation_wall_s - parallel - full,
                "full_database_validation": full,
            }
            if total is not None and llm_critical is not None
            else None
        ),
        accounting=(
            "Elapsed pipeline wall time sums construction and this invocation, excluding "
            "idle gaps and final timing-result writes. Parallel LLM attribution uses the "
            "last-finishing worker. Use paper_aggregate_total_s for paper comparisons. "
            "Cold-start totals are unavailable for resumed or uninstrumented runs."
        ),
    )
