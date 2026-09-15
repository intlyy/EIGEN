"""Summarize measured study artifacts without filling in absent experimental results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root: Path) -> dict:
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    calls = [
        json.loads(line)
        for path in root.glob("mini-*/tuning/llm/calls.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    known_costs = [c["cost_usd"] for c in calls if c.get("cost_usd") is not None]
    return {
        "status": result["status"],
        "best_candidate": result.get("best_candidate"),
        "best_metrics": result.get("best_metrics"),
        "timings": result.get("timings"),
        "local_evaluations": sum(
            r.get("completed_evaluations", 0) for r in result.get("local_results", [])
        ),
        "cross_validation_evaluations": result.get("cross_validation_evaluations", 0),
        "full_evaluations": result.get("full_validation_evaluations", 0),
        "llm_calls": len(calls),
        "reported_prompt_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) for c in calls),
        "reported_completion_tokens": sum(
            c.get("usage", {}).get("completion_tokens", 0) for c in calls
        ),
        "calls_without_usage": sum(not c.get("usage") for c in calls),
        "known_cost_subtotal_usd": sum(known_costs),
        "total_cost_usd": sum(known_costs) if len(known_costs) == len(calls) else None,
        "cost_note": "Configured token rates only; reconcile failed requests and pricing with provider billing.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.artifact_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
