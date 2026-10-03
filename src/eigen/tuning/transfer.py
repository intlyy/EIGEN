"""Transfer the complete feasible CALM frontier to cross-view validation."""

from __future__ import annotations

from collections.abc import Sequence

from eigen.config import TuningConfig

from .history import EvaluationRecord, candidate_key
from .pareto import archive


def transfer_pool(
    records: Sequence[EvaluationRecord], config: TuningConfig
) -> list[EvaluationRecord]:
    """Keep every non-dominated guidance point; never cap a region's frontier.

    The default coordinates are QPS and recall after applying all feasibility
    constraints. Final full-database ranking still maximizes feasible QPS.
    """
    selected = {
        candidate_key(r.candidate): r
        for r in archive(records, config.guidance_objectives(), config.all_constraints())
    }
    return list(selected.values())
