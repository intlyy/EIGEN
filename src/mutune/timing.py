"""Wall-clock accounting for construction and parallel study invocations."""

from __future__ import annotations

import math
import time


class ConstructionTimer:
    """Disjoint, sequential stages, including reads, writes and checksums."""

    def __init__(self):
        self.started = self.previous = time.monotonic()
        self.stages = {}

    def mark(self, name):
        now = time.monotonic()
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
    """Never add overlapping worker times or label a resumed run as cold start."""
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
    timings.update(
        total_invocation_wall_s=invocation_wall_s,
        construction_wall_s=construction,
        resumed_evaluations=resumed,
        cold_start=cold_start,
        cold_start_pipeline_wall_s=total,
        llm_worker_sum_s=sum(llm) if llm_known else None,
        llm_critical_path_wall_s=llm_critical,
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
            "Pipeline cost sums recorded construction and this study invocation, excluding "
            "idle gaps and final timing-result writes. Parallel LLM attribution uses the "
            "last-finishing worker; worker-sum LLM time is a non-additive diagnostic. "
            "Cold-start totals are unavailable for resumed or uninstrumented runs."
        ),
    )
