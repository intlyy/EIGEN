"""Command-line entry point for configuration-driven vector DB tuning."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from mutune import __version__
from mutune.api import EvaluationRequest, RunnerContext, WorkloadSpec
from mutune.config import LoadedProject, load_project
from mutune.errors import MuTuneError
from mutune.lifecycle import create_lifecycle
from mutune.llm import OpenAICompatibleClient
from mutune.plugins import create_runner, discover_runners, load_runner_class
from mutune.profiles import (
    list_builtin_profiles,
    load_profile,
    profile_fingerprint,
    validate_workload,
)
from mutune.rendering import ExperimentRenderer
from mutune.search_space import SearchSpace
from mutune.tuning import EvaluationRecord, Tuner
from mutune.utils import atomic_write_json, safe_name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mutune",
        description="Configuration-driven, plugin-extensible vector database tuning",
    )
    parser.add_argument("--version", action="version", version=f"muTune {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    study = commands.add_parser("study", help="parallel MiniDB tuning and full-database transfer")
    study.add_argument("config", type=Path)
    study.add_argument("--validate-only", action="store_true")
    source_digest = commands.add_parser(
        "benchmark-fingerprint", help="hash external benchmark source without executing it"
    )
    source_digest.add_argument("repo", type=Path)

    profiles = commands.add_parser("profiles", help="inspect engine profiles")
    profile_commands = profiles.add_subparsers(dest="profiles_command", required=True)
    profile_commands.add_parser("list", help="list built-in profile IDs")
    show_profile = profile_commands.add_parser("show", help="show one validated profile")
    show_profile.add_argument("profile", help="built-in profile ID or JSON path")

    plugins = commands.add_parser("plugins", help="inspect runner plugins")
    plugin_commands = plugins.add_subparsers(dest="plugins_command", required=True)
    plugin_commands.add_parser("list", help="list built-in and installed runners")

    validate = commands.add_parser("validate", help="validate a project without running it")
    validate.add_argument("config", type=Path)

    render = commands.add_parser("render", help="render one canonical candidate")
    render.add_argument("config", type=Path)
    render.add_argument(
        "--candidate",
        help="JSON object or path to a JSON object; profile defaults are used if omitted",
    )
    render.add_argument("--output", type=Path, help="also write the rendered JSON here")

    tune = commands.add_parser("tune", help="run or resume a tuning experiment")
    tune.add_argument("config", type=Path)
    tune.add_argument(
        "--dry-run",
        action="store_true",
        help="override the configured runner with the deterministic dry-run runner",
    )

    evaluate = commands.add_parser(
        "evaluate",
        help="evaluate one fixed candidate one or more times without tuning",
    )
    evaluate.add_argument("config", type=Path)
    evaluate_candidates = evaluate.add_mutually_exclusive_group()
    evaluate_candidates.add_argument(
        "--candidate",
        help="JSON object or path to a JSON object; profile defaults are used if omitted",
    )
    evaluate_candidates.add_argument(
        "--candidate-set",
        help="JSON array or path to a JSON array of candidate objects",
    )
    evaluate.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "number of repeated measurements; a state-reuse runner may retain "
            "compatible data and indexes (default: 1)"
        ),
    )
    evaluate.add_argument(
        "--dry-run",
        action="store_true",
        help="override the configured runner with the deterministic dry-run runner",
    )
    return parser


def _runtime(project: LoadedProject, *, experiment_name: str | None = None) -> dict[str, Any]:
    execution = project.config.execution
    available: dict[str, Any] = {
        "experiment_name": experiment_name or project.config.experiment_name,
        "connection_params": dict(execution.connection_params),
        "upload_parallel": execution.upload_parallel,
        "search_parallel": execution.search_parallel,
        "top_k": execution.top_k,
        "batch_size": execution.batch_size,
        "vector_size": execution.vector_size,
    }
    declared = set(project.profile.experiment.runtime_bindings) | set(
        project.profile.experiment.runtime_context
    )
    return {
        name: value for name, value in available.items() if name in declared and value is not None
    }


def _workload(project: LoadedProject) -> WorkloadSpec:
    execution = project.config.execution
    return WorkloadSpec(
        dataset=execution.dataset,
        distance=execution.distance,
        top_k=execution.top_k,
        concurrency=execution.search_parallel,
        vector_size=execution.vector_size,
        filtered=execution.filtered,
        sparse=execution.sparse,
    )


def _runner_id(project: LoadedProject) -> str:
    return project.config.runner.plugin or project.profile.adapter.plugin


def _read_candidate(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    candidate_path = Path(value).expanduser()
    try:
        raw = candidate_path.read_text(encoding="utf-8") if candidate_path.is_file() else value
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read candidate JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("candidate JSON must be an object")
    return payload


def _read_candidate_set(value: str) -> list[dict[str, Any]]:
    candidate_path = Path(value).expanduser()
    try:
        raw = candidate_path.read_text(encoding="utf-8") if candidate_path.is_file() else value
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read candidate-set JSON: {error}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError("candidate-set JSON must be a non-empty array")
    if not all(isinstance(candidate, dict) for candidate in payload):
        raise ValueError("every candidate-set entry must be an object")
    return payload


def _validate_project(project: LoadedProject) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_workload(project.profile, _workload(project))
    # Loading the selected class verifies the installed entry point, API version,
    # declared plugin ID, and importability without starting a database.
    load_runner_class(_runner_id(project))
    runtime = _runtime(project)
    candidate = SearchSpace(project.profile).canonicalize({}, runtime=runtime)
    rendered = ExperimentRenderer(project.profile).render(candidate, runtime=runtime)
    return candidate, rendered


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def _profiles_command(args: argparse.Namespace) -> int:
    if args.profiles_command == "list":
        for profile_id in list_builtin_profiles():
            print(profile_id)
        return 0
    profile = load_profile(args.profile)
    _print_json(profile.model_dump(mode="json"))
    return 0


def _plugins_command(_args: argparse.Namespace) -> int:
    for plugin_id, plugin in discover_runners().items():
        distribution = ""
        if plugin.distribution:
            distribution = f" [{plugin.distribution} {plugin.distribution_version or ''}]"
        print(f"{plugin_id}\t{plugin.source}{distribution}")
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    project = load_project(args.config)
    candidate, _rendered = _validate_project(project)
    _print_json(
        {
            "valid": True,
            "project": str(project.source_path),
            "profile": project.profile.id,
            "profile_fingerprint": profile_fingerprint(project.profile),
            "runner": _runner_id(project),
            "canonical_defaults": candidate,
            "artifact_dir": str(project.artifact_dir),
        }
    )
    return 0


def _render_command(args: argparse.Namespace) -> int:
    project = load_project(args.config)
    validate_workload(project.profile, _workload(project))
    runtime = _runtime(project)
    candidate = _read_candidate(args.candidate)
    rendered = ExperimentRenderer(project.profile).render(candidate, runtime=runtime)
    if args.output is not None:
        atomic_write_json(args.output.expanduser().resolve(), rendered)
    _print_json(rendered)
    return 0


def _merged_runner_settings(project: LoadedProject) -> dict[str, Any]:
    return {
        **dict(project.profile.adapter.options),
        **dict(project.config.runner.settings),
    }


def _server_params(rendered_experiment: dict[str, Any]) -> dict[str, Any]:
    value = rendered_experiment.get("server_params", {})
    if not isinstance(value, dict):
        raise ValueError("rendered server_params must be an object")
    return value


def _tune_command(args: argparse.Namespace) -> int:
    project = load_project(args.config)
    if args.dry_run:
        project.config.tuning.strategy = "random"
        project.config.llm = None
    validate_workload(project.profile, _workload(project))
    project.artifact_dir.mkdir(parents=True, exist_ok=True)

    execution = project.config.execution.model_dump(mode="json")
    lifecycle = create_lifecycle(
        project.config.lifecycle.model_dump(mode="json"),
        workspace_root=project.source_path.parent,
        artifact_dir=project.artifact_dir / "lifecycle",
        default_endpoint=str(execution.get("host", "localhost")),
        default_project_name=(
            "mutune-" + safe_name(project.config.experiment_name, 50).lower().replace(".", "-")
        ),
    )
    runner = None
    tuner_created = False
    try:
        # Runner construction is side-effect free.  The first candidate must be
        # rendered before an owned database can be started with its server
        # parameters, so use the stable configured endpoint here and start the
        # lifecycle from the before-evaluation hook below.
        execution["host"] = lifecycle.endpoint
        selected_runner = "dry-run" if args.dry_run else _runner_id(project)
        context = RunnerContext(
            profile=project.profile,
            settings=_merged_runner_settings(project),
            execution=execution,
            artifact_dir=project.artifact_dir,
        )
        runner = create_runner(selected_runner, context)

        lifecycle_started = False

        def before_evaluation(request: EvaluationRequest) -> None:
            nonlocal lifecycle_started
            if selected_runner == "dry-run":
                print(f"[muTune] preparing {request.run_id}; simulated evaluation", file=sys.stderr)
                return
            index_name = request.candidate.get(
                "index.type",
                request.candidate.get("index.mode", "default"),
            )
            print(
                f"[muTune] preparing {request.run_id} (index={index_name})",
                file=sys.stderr,
                flush=True,
            )
            changed = lifecycle.configure(
                request.engine_id,
                _server_params(dict(request.rendered_experiment)),
            )
            if not lifecycle_started:
                print("[muTune] starting database service...", file=sys.stderr, flush=True)
                execution["host"] = lifecycle.start()
                lifecycle_started = True
            elif changed or project.config.lifecycle.restart_between_evaluations:
                reason = "server parameters changed" if changed else "restart policy"
                print(
                    f"[muTune] restarting database service ({reason})...",
                    file=sys.stderr,
                    flush=True,
                )
                lifecycle.restart(
                    preserve_data=runner.can_reuse_database_state(),
                )
            print(
                "[muTune] database ready; running benchmark...",
                file=sys.stderr,
                flush=True,
            )

        def after_evaluation(record: EvaluationRecord) -> None:
            metrics = ", ".join(
                f"{name}={value:.6g}" for name, value in sorted(record.metrics.items())
            )
            suffix = f" ({metrics})" if metrics else ""
            print(
                f"[muTune] completed sequence={record.sequence} status={record.status}{suffix}",
                file=sys.stderr,
                flush=True,
            )

        llm_client = (
            OpenAICompatibleClient(project.config.llm) if project.config.llm is not None else None
        )
        tuner = Tuner(
            profile=project.profile,
            tuning=project.config.tuning,
            execution=execution,
            runner=runner,
            artifact_dir=project.artifact_dir,
            experiment_name=project.config.experiment_name,
            llm_client=llm_client,
            before_evaluation=before_evaluation,
            after_evaluation=after_evaluation,
            evaluation_timeout_s=float(
                project.config.runner.settings.get(
                    "timeout_s",
                    project.config.runner.settings.get("timeout", 86_400.0),
                )
            ),
            proposal_timeout_s=(
                project.config.llm.timeout_s if project.config.llm is not None else 120.0
            ),
        )
        tuner_created = True
        result = tuner.run()
    finally:
        if runner is not None and not tuner_created:
            runner.close()
        lifecycle.stop()

    _print_json(result.to_dict())
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")

    project = load_project(args.config)
    workload = _workload(project)
    validate_workload(project.profile, workload)
    project.artifact_dir.mkdir(parents=True, exist_ok=True)

    execution = project.config.execution.model_dump(mode="json")
    lifecycle = create_lifecycle(
        project.config.lifecycle.model_dump(mode="json"),
        workspace_root=project.source_path.parent,
        artifact_dir=project.artifact_dir / "lifecycle",
        default_endpoint=str(execution.get("host", "localhost")),
        default_project_name=(
            "mutune-" + safe_name(project.config.experiment_name, 50).lower().replace(".", "-")
        ),
    )
    runner = None
    observations: list[dict[str, Any]] = []
    canonical_candidates: list[dict[str, Any]] = []
    try:
        runtime = _runtime(project)
        raw_candidates = (
            _read_candidate_set(args.candidate_set)
            if args.candidate_set is not None
            else [_read_candidate(args.candidate)]
        )
        space = SearchSpace(project.profile)
        canonical_candidates = [
            space.canonicalize(candidate, runtime=runtime) for candidate in raw_candidates
        ]
        selected_runner = "dry-run" if args.dry_run else _runner_id(project)
        runner = create_runner(
            selected_runner,
            RunnerContext(
                profile=project.profile,
                settings=_merged_runner_settings(project),
                execution=execution,
                artifact_dir=project.artifact_dir,
            ),
        )
        lifecycle_started = False
        timeout_s = float(
            project.config.runner.settings.get(
                "timeout_s",
                project.config.runner.settings.get("timeout", 86_400.0),
            )
        )

        for candidate_index, canonical in enumerate(canonical_candidates, start=1):
            rendered = ExperimentRenderer(project.profile).render(
                canonical,
                runtime=runtime,
            )
            changed = (
                False
                if selected_runner == "dry-run"
                else lifecycle.configure(
                    project.profile.adapter.engine,
                    _server_params(rendered),
                )
            )
            if not lifecycle_started:
                execution["host"] = (
                    lifecycle.endpoint if selected_runner == "dry-run" else lifecycle.start()
                )
                lifecycle_started = True
            elif selected_runner != "dry-run" and (
                changed or project.config.lifecycle.restart_between_evaluations
            ):
                execution["host"] = lifecycle.restart(
                    preserve_data=runner.can_reuse_database_state(),
                )
            for repeat_index in range(1, args.repeat + 1):
                if (
                    repeat_index > 1
                    and selected_runner != "dry-run"
                    and project.config.lifecycle.restart_between_evaluations
                ):
                    execution["host"] = lifecycle.restart(
                        preserve_data=runner.can_reuse_database_state(),
                    )
                if len(canonical_candidates) == 1:
                    run_suffix = f"fixed-{repeat_index:03d}"
                else:
                    run_suffix = f"candidate-{candidate_index:03d}-repeat-{repeat_index:03d}"
                run_id = safe_name(f"{project.config.experiment_name}-{run_suffix}")
                observation = runner.evaluate(
                    EvaluationRequest(
                        run_id=run_id,
                        engine_id=project.profile.adapter.engine,
                        candidate=canonical,
                        rendered_experiment=rendered,
                        workload=workload,
                        seed=(project.config.tuning.seed + len(observations)),
                        timeout_s=timeout_s,
                    )
                )
                observations.append(
                    {
                        "candidate_index": candidate_index,
                        "candidate": canonical,
                        "repeat": repeat_index,
                        "run_id": run_id,
                        "status": observation.status.value,
                        "metrics": dict(observation.metrics),
                        "auxiliary": dict(observation.auxiliary),
                        "artifacts": list(observation.artifacts),
                        "error": observation.error,
                    }
                )
    finally:
        try:
            if runner is not None:
                runner.close()
        finally:
            lifecycle.stop()

    successful = [item for item in observations if item["status"] == "ok"]
    metric_names = sorted(
        set.intersection(*(set(item["metrics"]) for item in successful)) if successful else set()
    )
    means = {
        name: statistics.fmean(float(item["metrics"][name]) for item in successful)
        for name in metric_names
    }
    candidate_summaries: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(canonical_candidates, start=1):
        candidate_observations = [
            item for item in observations if item["candidate_index"] == candidate_index
        ]
        candidate_successes = [item for item in candidate_observations if item["status"] == "ok"]
        names = sorted(
            set.intersection(*(set(item["metrics"]) for item in candidate_successes))
            if candidate_successes
            else set()
        )
        candidate_summaries.append(
            {
                "candidate_index": candidate_index,
                "candidate": candidate,
                "requested_repeats": args.repeat,
                "successful_repeats": len(candidate_successes),
                "mean_metrics": {
                    name: statistics.fmean(
                        float(item["metrics"][name]) for item in candidate_successes
                    )
                    for name in names
                },
            }
        )
    result: dict[str, Any] = {
        "requested_candidates": len(canonical_candidates),
        "requested_repeats_per_candidate": args.repeat,
        "successful_evaluations": len(successful),
        "candidate_summaries": candidate_summaries,
        "observations": observations,
    }
    if len(canonical_candidates) == 1:
        result.update(
            {
                "candidate": canonical_candidates[0],
                "requested_repeats": args.repeat,
                "successful_repeats": len(successful),
                "mean_metrics": means,
            }
        )
    _print_json(result)
    expected = len(canonical_candidates) * args.repeat
    return 0 if len(successful) == expected else 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "benchmark-fingerprint":
        from mutune.runners.vectordb_benchmark import benchmark_source_sha256

        print(benchmark_source_sha256(args.repo.resolve()))
        return 0
    if args.command == "study":
        from mutune.study import load_study, run_study

        study = load_study(args.config)
        if args.validate_only:
            for project in [*study.minidbs, study.full_database]:
                _validate_project(project)
            _print_json(
                {
                    "valid": True,
                    "minidbs": len(study.minidbs),
                    "artifact_dir": str(study.artifact_dir),
                }
            )
            return 0
        result = run_study(study)
        _print_json(result)
        return 0 if result["status"] == "ok" else 1
    if args.command == "profiles":
        return _profiles_command(args)
    if args.command == "plugins":
        return _plugins_command(args)
    if args.command == "validate":
        return _validate_command(args)
    if args.command == "render":
        return _render_command(args)
    if args.command == "tune":
        return _tune_command(args)
    if args.command == "evaluate":
        return _evaluate_command(args)
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except KeyboardInterrupt:
        print("mutune: interrupted", file=sys.stderr)
        return 130
    except (MuTuneError, OSError, TypeError, ValueError) as error:
        print(f"mutune: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
