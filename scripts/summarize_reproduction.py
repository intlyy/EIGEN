"""Export measured experiment cells to JSON and CSV; missing values remain missing."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from scripts.summarize_study import summarize
except ModuleNotFoundError:
    from summarize_study import summarize


def measured_number(value):
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        else None
    )


def summarize_plan(plan_path: Path) -> list[dict]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = []
    for run in plan["runs"]:
        row = {
            key: run.get(key)
            for key in (
                "id",
                "dataset",
                "engine",
                "method",
                "manifest_label",
                "num_minidbs",
                "sample_ratio",
                "recall_threshold",
                "seed",
                "backbone",
                "artifact_dir",
            )
        }
        row.update(
            status="missing",
            metrics_origin=None,
            qps=None,
            recall=None,
            feasible=None,
            paper_aggregate_total_s=None,
            paper_components_s=None,
            invocation_wall_s=None,
            local_evaluations=None,
            cross_validation_evaluations=None,
            full_evaluations=None,
            total_cost_usd=None,
        )
        root = Path(run["artifact_dir"])
        if not root.is_absolute():
            root = plan_path.parent / root
        result_path = root / "result.json"
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(result, dict):
                    raise ValueError("result.json must contain an object")
                if result.get("best_metrics") is not None and not isinstance(
                    result["best_metrics"], dict
                ):
                    raise ValueError("best_metrics must contain an object or null")
                row["metrics_origin"] = result.get("metrics_origin", "unknown_legacy")
                if run["method"] == "direct-full":
                    row.update(
                        status=result.get("status", "unknown_legacy"),
                        local_evaluations=result.get("completed_evaluations"),
                        invocation_wall_s=measured_number(result.get("invocation_wall_s")),
                    )
                    if (
                        result.get("status") == "ok"
                        and result.get("complete") is True
                        and type(result.get("resumed_evaluations")) is int
                        and result["resumed_evaluations"] == 0
                    ):
                        row["paper_aggregate_total_s"] = row["invocation_wall_s"]
                else:
                    summary = summarize(root)
                    for key in (
                        "status",
                        "paper_aggregate_total_s",
                        "paper_components_s",
                        "local_evaluations",
                        "cross_validation_evaluations",
                        "full_evaluations",
                        "total_cost_usd",
                    ):
                        row[key] = summary.get(key)
                if row["metrics_origin"] == "synthetic":
                    row.update(
                        status="synthetic", paper_aggregate_total_s=None, paper_components_s=None
                    )
                metrics = (result.get("best_metrics") or {}) if row["status"] == "ok" else {}
                row["qps"] = measured_number(metrics.get("qps"))
                row["recall"] = measured_number(metrics.get("recall"))
                if row["recall"] is not None and row["qps"] is not None:
                    row["feasible"] = row["recall"] >= run["recall_threshold"]
            except (OSError, ValueError, TypeError, KeyError) as error:
                row.update(
                    status="invalid_artifact",
                    error=str(error),
                    qps=None,
                    recall=None,
                    feasible=None,
                    paper_aggregate_total_s=None,
                    paper_components_s=None,
                    total_cost_usd=None,
                )
        rows.append(row)
    return rows


def write_summary(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                    for key, value in row.items()
                }
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = summarize_plan(args.plan.resolve())
    write_summary(rows, args.output.resolve())
    print(
        f"Summarized {len(rows)} cells; {sum(row['status'] == 'missing' for row in rows)} missing."
    )


if __name__ == "__main__":
    main()
