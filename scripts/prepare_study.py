"""Generate portable project configs from a completed MiniDB manifest (no DB/API calls)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from mutune.runners.vectordb_benchmark import benchmark_source_sha256
from mutune.utils import atomic_write_json


def project_payload(
    engine,
    dataset_path,
    *,
    label,
    port,
    dimension,
    distance,
    benchmark_repo,
    python_executable,
    seed,
    model,
    top_k=10,
):
    """Repository defaults; unspecified paper hyperparameters are documented."""
    connection = {"port": port}
    host = "127.0.0.1"
    environment = {"MUTUNE_PORT": str(port)}
    if engine == "qdrant":
        host = f"http://127.0.0.1:{port}"
        connection = {"grpc_port": port + 1, "timeout": 120}
        environment["MUTUNE_GRPC_PORT"] = str(port + 1)
    elif engine == "pgvector":
        connection.update(dbname="postgres", user="postgres")
    else:
        environment["MUTUNE_HTTP_PORT"] = str(9091 + port - 19530)
    lifecycle = {
        "mode": "docker_compose",
        "settings": {
            "compose_file": f"./deploy/{engine}.compose.yml",
            "project_name": f"mutune-{engine}-{label}",
            "endpoint": host,
            "environment": environment,
            "ready_check": {"kind": "tcp", "host": "127.0.0.1", "port": port, "timeout_s": 180},
            "command_timeout_s": 300,
            "remove_volumes": True,
        },
    }
    if engine != "qdrant":
        lifecycle["settings"]["server_config"] = {
            "kind": "postgresql" if engine == "pgvector" else "milvus_user_yaml",
            "service": engine,
        }
    return {
        "schema_version": 1,
        "experiment_name": f"{engine}-{label}",
        "engine_profile": f"{engine}-{'ann' if engine == 'pgvector' else 'native'}-dense",
        "artifact_dir": f"./artifacts/{engine}-{label}",
        "execution": {
            "dataset": f"{engine}-{label}",
            "dataset_description": "Set the dataset name and characteristics before experiments.",
            "host": host,
            "connection_params": connection,
            "distance": distance,
            "vector_size": dimension,
            "top_k": top_k,
            "search_parallel": 16,
            "upload_parallel": 16,
            "batch_size": 1024,
            "hardware": {"cpu_cores_per_database": 16, "memory_gib_per_database": 64},
        },
        "runner": {
            "plugin": "vector-db-benchmark",
            "settings": {
                "repo_path": str(benchmark_repo),
                "python_executable": str(python_executable),
                "dataset_path": str(dataset_path),
                "state_reuse": False,
                "keep_workspace": True,
                "timeout_s": 86400,
                "pass_env": ["PGVECTOR_PASSWORD"],
            },
        },
        "lifecycle": lifecycle,
        "tuning": {
            "strategy": "calm",
            "budget": 60,
            "initial_samples": 14,
            "proposals_per_round": 12,
            "evaluations_per_round": 4,
            "regions_per_round": 1,
            "seed": seed,
            "recall_threshold": 0.9,
            "objectives": [
                {"metric": "qps", "direction": "maximize"},
            ],
        },
        "llm": {
            "model": model,
            "temperature": None,
            "max_tokens": None,
            "extra_body": {"reasoning_effort": "medium"},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["milvus", "qdrant", "pgvector"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--benchmark-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") == "geo" and args.engine != "qdrant":
        parser.error("use Qdrant for the bundled geo-radius adapter")
    if not 0 < args.top_k <= manifest["top_k"]:
        parser.error("--top-k exceeds manifest ground truth or is nonpositive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("choose an empty output directory to preserve existing experiment configs")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "examples/paper/deploy", output / "deploy"
    )
    base_port = {"milvus": 19530, "qdrant": 6333, "pgvector": 5432}[args.engine]
    names = []
    source_digest = benchmark_source_sha256(args.benchmark_repo.resolve())
    entries = [*manifest["minidbs"], {"path": manifest["source"], "id": "full"}]
    for i, entry in enumerate(entries):
        full = i == len(entries) - 1
        label = "full" if full else f"mini-{i:02d}"
        dataset_path = Path(entry["path"])
        if not dataset_path.is_absolute():
            dataset_path = manifest_path.parent / dataset_path
        payload = project_payload(
            args.engine,
            dataset_path.resolve(),
            label=label,
            port=base_port + (0 if full else i * 2),
            dimension=manifest["dimension"],
            distance=manifest["metric"],
            benchmark_repo=args.benchmark_repo.resolve(),
            python_executable=args.benchmark_python.resolve(),
            seed=42 + i,
            model=args.model,
            top_k=args.top_k,
        )
        payload["runner"]["settings"]["expected_source_sha256"] = source_digest
        if manifest.get("format") == "geo":
            payload["execution"]["filtered"] = True
            payload["runner"]["settings"]["dataset_entry"] = {"schema": manifest["payload_schema"]}
        name = f"{args.engine}-{label}.json"
        atomic_write_json(output / name, payload)
        names.append(name)
    atomic_write_json(
        output / "study.json",
        {
            "name": f"{args.engine}-study",
            "minidbs": names[:-1],
            "full_database": names[-1],
            "minidb_manifest": str(manifest_path),
            "artifact_dir": "./artifacts/study",
            "top_l": 5,
            "stability_weight": 1.0,
        },
    )
    print(output / "study.json")


if __name__ == "__main__":
    main()
