"""Feasible transfer representatives, separate from the optimizer's frontier."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from mutune.config import TuningConfig

from .history import EvaluationRecord, candidate_key
from .pareto import archive, feasible


def transfer_pool(
    records: Sequence[EvaluationRecord], config: TuningConfig
) -> list[EvaluationRecord]:
    """Union the true frontier with <= M representatives per structural region.

    Always reserve the fastest feasible and highest-recall candidates, then
    fill remaining region slots by primary objective. Recall remains a
    constraint during optimization; this is a transfer selection policy.
    """
    selected = {
        candidate_key(r.candidate): r
        for r in archive(records, config.objectives, config.all_constraints())
    }
    regions = defaultdict(list)
    for record in records:
        if (
            record.ok
            and feasible(record.metrics, config.all_constraints())
            and all(
                type(record.metrics.get(o.metric)) in (int, float)
                and math.isfinite(record.metrics[o.metric])
                for o in config.objectives
            )
        ):
            regions[record.region_id].append(record)
    for records_in_region in regions.values():
        ordered = sorted(
            records_in_region, key=lambda r: (-r.metrics[config.objective_metric], r.sequence)
        )
        safest = min(
            ordered,
            key=lambda r: (
                -r.metrics[config.constraint_metric],
                -r.metrics[config.objective_metric],
                r.sequence,
            ),
        )
        representatives = {}
        for record in [ordered[0], safest, *ordered]:
            representatives.setdefault(candidate_key(record.candidate), record)
            if len(representatives) == config.transfer_candidates_per_region:
                break
        selected.update(representatives)
    return [selected[key] for key in sorted(selected)]
