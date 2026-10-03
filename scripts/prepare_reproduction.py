"""Generate a reviewable experiment matrix from existing, independently built manifests.

No database, model API, data download, or benchmark process is started here.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import shutil
import sys
from pathlib import Path

from eigen.benchmark_compat import MILVUS_GEO_CONTRACT
from eigen.config import LLMConfig, ProjectConfig
from eigen.runners.vectordb_benchmark import benchmark_source_sha256
from eigen.study import StudyConfig
from eigen.utils import atomic_write_json

try:
    from scripts.prepare_study import project_payload
except ModuleNotFoundError:  # Direct `python scripts/prepare_reproduction.py` invocation.
    from prepare_study import project_payload

PAPER_RECALLS = (0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 0.99)
PAPER_DATASETS = {"nytimes", "glove", "gist", "tiny5m", "msong", "geo-radius"}
METHODS = ("eigen", "mean-only", "uniform-mini", "direct-full")
PAPER_ALLOCATION_VERSION = "minimum-one-original-population-largest-remainder-v2"


def portable_label(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", value):
        raise ValueError(f"labels require 1-40 letters, digits, underscores or hyphens: {value!r}")
    return value


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def is_uniform(manifest: dict) -> bool:
    return manifest.get("sampling_method") == "uniform"


def build_matrix(args) -> dict:
    inventory_path = args.inventory.resolve()
    inventory = read_json(inventory_path)
    datasets = inventory["datasets"]
    names = [portable_label(dataset["name"]) for dataset in datasets]
    if not datasets or len(set(names)) != len(names):
        raise ValueError("datasets must be nonempty and have distinct names")
    if args.require_paper_datasets and set(names) != PAPER_DATASETS:
        raise ValueError(
            "--require-paper-datasets requires exactly the six names in the example inventory"
        )
    for values, name in (
        (args.recalls, "recalls"),
        (args.seeds, "seeds"),
        (args.engines, "engines"),
        (args.methods, "methods"),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must not contain duplicates")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in args.recalls):
        raise ValueError("recall thresholds must be finite values in [0, 1]")
    if "direct-full" in args.methods and (args.direct_full_budget or 0) < 1:
        raise ValueError("direct-full requires an explicit positive --direct-full-budget")
    if (
        min(
            args.budget_per_minidb,
            args.top_k,
            args.top_l,
            args.search_parallel,
            args.upload_parallel,
        )
        < 1
    ):
        raise ValueError("budgets, top-k, top-l and concurrency must be positive")
    backbones = (
        read_json(args.backbones.resolve())["backbones"]
        if args.backbones
        else [
            {
                "name": "gpt-5_4-medium",
                "llm": {
                    "model": "gpt-5.4",
                    "temperature": None,
                    "max_tokens": None,
                    "extra_body": {"reasoning_effort": "medium"},
                },
            }
        ]
    )
    backbone_names = [portable_label(item["name"]) for item in backbones]
    if not backbones or len(set(backbone_names)) != len(backbone_names):
        raise ValueError("backbones must be nonempty and have distinct names")
    for item in backbones:
        LLMConfig.model_validate(item["llm"])
    benchmark = args.benchmark_repo.resolve()
    source_digest = benchmark_source_sha256(benchmark)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "choose an empty output directory; existing experiments are never overwritten"
        )

    # Validate the whole input before producing any project files.
    prepared = []
    for dataset in datasets:
        if not isinstance(dataset.get("description"), str) or not dataset["description"].strip():
            raise ValueError(f"{dataset['name']}: provide a real dataset description")
        variants = dataset["manifests"]
        labels = [portable_label(variant["label"]) for variant in variants]
        if not variants or len(set(labels)) != len(labels):
            raise ValueError(f"{dataset['name']}: manifest labels must be nonempty and distinct")
        for variant in variants:
            path = dataset_path(inventory_path, variant["path"])
            manifest = read_json(path)
            if manifest.get("sampling_method") not in {"locality-stratified", "uniform"}:
                raise ValueError(
                    f"{path}: unknown sampling method; rebuild Mini-DBs with this release"
                )
            if (
                not is_uniform(manifest)
                and manifest.get("allocation_version") != PAPER_ALLOCATION_VERSION
            ):
                raise ValueError(
                    f"{path}: old/unknown allocation version; rebuild Mini-DBs with this release"
                )
            if args.top_k > manifest["top_k"]:
                raise ValueError(f"{path}: requested top-k exceeds ground truth")
            if not manifest.get("minidbs") or not manifest.get("source_sha256"):
                raise ValueError(f"{path}: incomplete Mini-DB manifest")
            if (
                type(manifest.get("source_size")) is not int
                or manifest["source_size"] < 1
                or type(manifest.get("target_size")) is not int
                or not 0 < manifest["target_size"] <= manifest["source_size"]
            ):
                raise ValueError(f"{path}: invalid source/target cardinalities")
            if not dataset_path(path, manifest["source"]).exists():
                raise ValueError(f"{path}: source data are missing")
            for entry in manifest["minidbs"]:
                if not dataset_path(path, entry["path"]).exists() or not entry.get("sha256"):
                    raise ValueError(f"{path}: Mini-DB data or checksum are missing")
            prepared.append((dataset, variant, path, manifest))
        for method in args.methods:
            if method == "uniform-mini" and not any(
                is_uniform(manifest) for item, _, _, manifest in prepared if item is dataset
            ):
                raise ValueError(
                    f"{dataset['name']}: uniform-mini needs a separately built uniform manifest"
                )
            if method != "uniform-mini" and not any(
                not is_uniform(manifest) for item, _, _, manifest in prepared if item is dataset
            ):
                raise ValueError(f"{dataset['name']}: {method} needs a stratified manifest")

    runs, skipped = [], []
    for dataset, variant, manifest_path, manifest in prepared:
        for engine, recall, seed, backbone, method in itertools.product(
            args.engines, args.recalls, args.seeds, backbones, args.methods
        ):
            if (method == "uniform-mini") != is_uniform(manifest):
                continue
            if method == "direct-full":
                # One Direct-Full run per dataset, independent of Mini-DB sensitivity variants.
                first_stratified = next(
                    v for d, v, _, m in prepared if d is dataset and not is_uniform(m)
                )
                if variant is not first_stratified:
                    continue
            if manifest.get("format") == "geo" and engine == "pgvector":
                skipped.append(
                    {
                        "dataset": dataset["name"],
                        "engine": engine,
                        "reason": "bundled pgvector adapter does not support geo filtering",
                    }
                )
                continue
            tag = f"r{recall}-s{seed}"
            run_id = "/".join(
                [dataset["name"], engine, method, variant["label"], backbone["name"], tag]
            )
            directory = output / run_id
            directory.mkdir(parents=True)
            shutil.copytree(
                Path(__file__).resolve().parents[1] / "examples/paper/deploy", directory / "deploy"
            )
            projects = []
            entries = (
                [*manifest["minidbs"], {"path": manifest["source"], "id": "full"}]
                if method != "direct-full"
                else [{"path": manifest["source"], "id": "full"}]
            )
            for index, entry in enumerate(entries):
                full = index == len(entries) - 1
                label = "full" if full else f"mini-{index:02d}"
                base_port = {"milvus": 19530, "qdrant": 6333, "pgvector": 5432}[engine]
                payload = project_payload(
                    engine,
                    dataset_path(manifest_path, entry["path"]),
                    label=label,
                    port=base_port + (0 if full else index * 2),
                    dimension=manifest["dimension"],
                    distance=manifest["metric"],
                    benchmark_repo=benchmark,
                    python_executable=args.benchmark_python.resolve(),
                    seed=seed + index,
                    model=backbone["llm"]["model"],
                    top_k=args.top_k,
                    budget=args.direct_full_budget
                    if method == "direct-full"
                    else args.budget_per_minidb,
                )
                payload["execution"].update(
                    dataset_description=dataset["description"],
                    search_parallel=args.search_parallel,
                    upload_parallel=args.upload_parallel,
                )
                payload["tuning"].update(recall_threshold=recall, initial_samples=None)
                payload["llm"] = backbone["llm"]
                settings = payload["runner"]["settings"]
                settings.update(
                    expected_source_sha256=source_digest,
                    expected_dataset_sha256=manifest["source_sha256"] if full else entry["sha256"],
                )
                if manifest.get("format") == "geo":
                    payload["execution"]["filtered"] = True
                    settings["dataset_entry"] = {"schema": manifest["payload_schema"]}
                    if engine == "milvus":
                        settings["milvus_geo_filter"] = MILVUS_GEO_CONTRACT
                short_id = hashlib.sha256(run_id.encode()).hexdigest()[:10]
                payload["lifecycle"]["settings"]["project_name"] = f"eigen-{short_id}-{label}"
                if method == "direct-full":
                    payload["artifact_dir"] = "./artifacts/direct-full"
                ProjectConfig.model_validate(payload)
                filename = f"{engine}-{label}.json"
                atomic_write_json(directory / filename, payload)
                projects.append(filename)
            if method == "direct-full":
                config = directory / projects[-1]
                artifact = directory / "artifacts/direct-full"
                command = [sys.executable, "-m", "eigen", "tune", str(config)]
                validation = [sys.executable, "-m", "eigen", "validate", str(config)]
            else:
                config = directory / "study.json"
                artifact = directory / "artifacts/study"
                study = {
                    "name": f"eigen-{dataset['name']}-{engine}",
                    "minidbs": projects[:-1],
                    "full_database": projects[-1],
                    "minidb_manifest": str(manifest_path),
                    "artifact_dir": "./artifacts/study",
                    "top_l": args.top_l,
                    "stability_weight": 0.0 if method == "mean-only" else 1.0,
                }
                StudyConfig.model_validate(study)
                atomic_write_json(config, study)
                command = [sys.executable, "-m", "eigen", "study", str(config)]
                validation = [*command, "--validate-only"]
            runs.append(
                {
                    "id": run_id,
                    "dataset": dataset["name"],
                    "engine": engine,
                    "method": method,
                    "manifest_label": variant["label"],
                    "manifest": str(manifest_path),
                    "sampling_method": manifest["sampling_method"],
                    "construction_method": manifest.get("method"),
                    "allocation_version": manifest.get("allocation_version"),
                    "source_sha256": manifest["source_sha256"],
                    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "num_minidbs": len(manifest["minidbs"]) if method != "direct-full" else 0,
                    "sample_ratio": manifest.get("target_size", 0) / manifest["source_size"]
                    if method != "direct-full"
                    else None,
                    "recall_threshold": recall,
                    "seed": seed,
                    "backbone": backbone["name"],
                    "config": str(config),
                    "artifact_dir": str(artifact),
                    "validate_argv": validation,
                    "run_argv": command,
                }
            )
    if not runs:
        raise ValueError("no supported experiment cells were selected")
    plan = {
        "schema_version": 1,
        "inventory": str(inventory_path),
        "benchmark_source_sha256": source_digest,
        "runs": runs,
        "skipped": skipped,
        "execution_policy": "Run cells serially: endpoint ports are reused between cells. Review resource placement first.",
        "results_status": "not_executed",
        "baselines_included": False,
    }
    atomic_write_json(output / "run-plan.json", plan)
    lines = [
        "# Reviewable reproduction commands",
        "",
        plan["execution_policy"],
        "",
        "Arguments are JSON arrays, not shell-escaped strings. Use subprocess.run(argv, check=True)",
        "only after reviewing the generated configs. No experiments have run.",
        "",
    ]
    for run in runs:
        lines.extend(
            [
                f"## {run['id']}",
                "",
                "```json",
                json.dumps(run["validate_argv"]),
                json.dumps(run["run_argv"]),
                "```",
                "",
            ]
        )
    (output / "COMMANDS.md").write_text("\n".join(lines), encoding="utf-8")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--benchmark-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--engines", nargs="+", choices=["milvus", "qdrant", "pgvector"], default=["milvus"]
    )
    parser.add_argument("--recalls", type=float, nargs="+", default=list(PAPER_RECALLS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["eigen"])
    parser.add_argument("--backbones", type=Path)
    parser.add_argument("--budget-per-minidb", type=int, default=20)
    parser.add_argument("--direct-full-budget", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-l", type=int, default=5)
    parser.add_argument("--search-parallel", type=int, default=16)
    parser.add_argument("--upload-parallel", type=int, default=16)
    parser.add_argument("--require-paper-datasets", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_matrix(args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Prepared {len(plan['runs'])} runs; executed 0. Review {args.output / 'run-plan.json'}")


if __name__ == "__main__":
    main()
