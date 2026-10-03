"""Summarize measured study artifacts without filling in absent experimental results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def summarize(root: Path) -> dict:
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("result.json must contain an object")
    local_results = result.get("local_results", [])
    if not isinstance(local_results, list) or any(
        not isinstance(row, dict) for row in local_results
    ):
        raise ValueError("local_results must contain worker result objects")
    worker_calls = {}
    for path in root.glob("mini-*/tuning/llm/calls.jsonl"):
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(
            not isinstance(row, dict) or not isinstance(row.get("usage", {}), dict)
            for row in records
        ):
            raise ValueError(f"invalid LLM call records: {path}")
        worker_calls[path.relative_to(root).parts[0]] = records
    calls = [call for records in worker_calls.values() for call in records]
    known_costs = [c["cost_usd"] for c in calls if c.get("cost_usd") is not None]
    if any(
        type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
        for cost in known_costs
    ):
        raise ValueError("invalid recorded LLM costs")
    # Missing files cannot establish a zero-dollar total. A recorded zero LLM
    # duration in every fresh worker can establish that no calls took place.
    logs_complete = bool(local_results) and all(
        type(row.get("resumed_evaluations")) is int
        and row["resumed_evaluations"] == 0
        and type(row.get("llm_wall_s")) in (int, float)
        and math.isfinite(row["llm_wall_s"])
        and row["llm_wall_s"] >= 0
        and (row["llm_wall_s"] == 0 or bool(worker_calls.get(f"mini-{i:02d}")))
        for i, row in enumerate(local_results)
    )
    timings = result.get("timings", {})
    return {
        "status": result["status"],
        "metrics_origin": result.get("metrics_origin", "unknown"),
        "best_candidate": result.get("best_candidate"),
        "best_metrics": result.get("best_metrics"),
        "timings": timings,
        "paper_aggregate_total_s": timings.get("paper_aggregate_total_s"),
        "paper_components_s": timings.get("paper_components_s"),
        "local_evaluations": sum(
            r.get("completed_evaluations", 0) for r in result.get("local_results", [])
        ),
        "cross_validation_evaluations": result.get("cross_validation_evaluations", 0),
        "full_evaluations": result.get("full_validation_evaluations", 0),
        "llm_calls": len(calls),
        "llm_logs_complete": logs_complete,
        "reported_prompt_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) for c in calls),
        "reported_completion_tokens": sum(
            c.get("usage", {}).get("completion_tokens", 0) for c in calls
        ),
        "calls_without_usage": sum(not c.get("usage") for c in calls),
        "known_cost_subtotal_usd": sum(known_costs),
        "total_cost_usd": sum(known_costs)
        if logs_complete and len(known_costs) == len(calls)
        else None,
        "cost_note": "Configured token rates only; reconcile failed requests and pricing with provider billing.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.artifact_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
