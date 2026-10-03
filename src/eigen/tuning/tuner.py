"""Sequential, budget-exact vector database tuning orchestration."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from eigen.api import (
    BaseRunner,
    EvaluationRequest,
    Observation,
    RunStatus,
    WorkloadSpec,
)
from eigen.config import ExecutionConfig, LoadedProject, TuningConfig
from eigen.errors import EIGENError
from eigen.llm import CompletionClient, LoggedCompletionClient
from eigen.models import EngineProfile
from eigen.profiles import profile_fingerprint, validate_workload
from eigen.rendering import ExperimentRenderer
from eigen.search_space import SearchSpace
from eigen.utils import atomic_write_json, fingerprint, safe_name

from .calm_selection import CALMScore, ParetoBatchSelector
from .history import EvaluationRecord, HistoryStore, candidate_key
from .llm_surrogate import LLMSurrogate
from .pareto import feasible
from .partitioning import ProfilePartitioner, Region
from .proposer import (
    RANDOM_FALLBACK_TIMEOUT_S,
    CandidateProposer,
    HybridProposer,
    OpenAICompatibleProposer,
    RandomProposer,
)
from .selection import AcquisitionScore, ConstraintAwareAcquisition
from .surrogate import MixedSpaceKnnSurrogate
from .transfer import transfer_pool


class TuningError(EIGENError):
    """Raised when a tuning run cannot safely make further progress."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _model_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected a Pydantic model or mapping, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class TuningResult:
    history_path: Path
    checkpoint_path: Path
    budget: int
    completed_evaluations: int
    best_candidate: dict[str, Any] | None
    best_metrics: dict[str, float] | None
    pareto_candidates: list[dict[str, Any]] = field(default_factory=list)
    transfer_candidates: list[dict[str, Any]] = field(default_factory=list)
    resumed_evaluations: int = 0
    llm_wall_s: float = 0.0

    @property
    def complete(self) -> bool:
        return self.completed_evaluations == self.budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_path": str(self.history_path),
            "checkpoint_path": str(self.checkpoint_path),
            "budget": self.budget,
            "completed_evaluations": self.completed_evaluations,
            "complete": self.complete,
            "best_candidate": self.best_candidate,
            "best_metrics": self.best_metrics,
            "pareto_candidates": self.pareto_candidates,
            "transfer_candidates": self.transfer_candidates,
            "resumed_evaluations": self.resumed_evaluations,
            "llm_wall_s": self.llm_wall_s,
        }


class Tuner:
    """Tune one MiniDB with CALM; the outer study parallelizes independent DBs.

    Parameters are intentionally explicit so CLI composition remains separate
    from optimizer behavior.  ``runner`` is owned by the tuner and is closed at
    the end of :meth:`run`, including on failure.
    """

    def __init__(
        self,
        *,
        profile: EngineProfile,
        tuning: TuningConfig,
        execution: ExecutionConfig | Mapping[str, Any],
        runner: BaseRunner,
        artifact_dir: str | Path,
        experiment_name: str,
        proposer: CandidateProposer | None = None,
        llm_client: CompletionClient | None = None,
        before_evaluation: Callable[[EvaluationRequest], None] | None = None,
        after_evaluation: Callable[[EvaluationRecord], None] | None = None,
        evaluation_timeout_s: float = 86_400.0,
        proposal_timeout_s: float = 120.0,
    ) -> None:
        if evaluation_timeout_s <= 0 or not math.isfinite(evaluation_timeout_s):
            raise ValueError("evaluation_timeout_s must be positive and finite")
        if proposal_timeout_s <= 0 or not math.isfinite(proposal_timeout_s):
            raise ValueError("proposal_timeout_s must be positive and finite")

        self.profile = profile
        self.tuning = tuning
        self.execution = _model_dict(execution)
        self.runner = runner
        self.before_evaluation = before_evaluation
        self.after_evaluation = after_evaluation
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = safe_name(experiment_name)
        self.evaluation_timeout_s = evaluation_timeout_s
        self.proposal_timeout_s = proposal_timeout_s
        self.rng = random.Random(tuning.seed)

        self.runtime = self._runtime_values()
        self.workload = self._workload()
        validate_workload(profile, self.workload)
        self.search_space = SearchSpace(profile, runtime_defaults=self.runtime)
        self.renderer = ExperimentRenderer(profile)
        self.partitioner = ProfilePartitioner(
            self.search_space,
            runtime=self.runtime,
        )
        self.random_proposer = RandomProposer(
            self.search_space,
            rng=self.rng,
            runtime=self.runtime,
        )
        self.task = {
            "dataset": self.workload.dataset,
            "distance": self.workload.distance,
            "vector_size": self.workload.vector_size,
            "top_k": self.workload.top_k,
            "concurrency": self.workload.concurrency,
            "filtered": self.workload.filtered,
            "hardware": self.execution.get("hardware", {}),
            "description": self.execution.get("dataset_description", ""),
        }
        proposal_client = (
            LoggedCompletionClient(llm_client, self.artifact_dir / "llm", "proposer")
            if llm_client is not None
            else None
        )
        self.proposer = proposer or self._default_proposer(proposal_client)
        self.logged_clients = [proposal_client] if proposal_client is not None else []
        self.llm_contract = (
            getattr(llm_client, "config", None).model_dump(mode="json")
            if getattr(llm_client, "config", None) is not None
            else None
        )
        if tuning.strategy == "calm":
            if llm_client is None:
                raise TuningError("CALM requires an LLM client for generation and prediction")
            if tuning.budget < len(self.partitioner.regions):
                raise TuningError("CALM budget must allow at least one seed per region")
            surrogate_client = LoggedCompletionClient(
                llm_client, self.artifact_dir / "llm", "surrogate"
            )
            self.logged_clients.append(surrogate_client)
            self.surrogate = LLMSurrogate(
                self.search_space,
                surrogate_client,
                objectives=tuning.objectives,
                guidance_objectives=tuning.guidance_objectives(),
                constraints=tuning.all_constraints(),
                task=self.task,
                history_limit=tuning.history_limit,
                batch_size=tuning.surrogate_batch_size,
            )
        else:
            self.surrogate = MixedSpaceKnnSurrogate(
                self.search_space,
                objective_metric=tuning.objective_metric,
                constraint_metric=tuning.constraint_metric,
            )
        self.acquisition = ConstraintAwareAcquisition(
            objective_metric=tuning.objective_metric,
            constraint_metric=tuning.constraint_metric,
            constraint_threshold=tuning.recall_threshold,
            exploration_weight=tuning.exploration_weight,
        )
        self.calm_acquisition = ParetoBatchSelector(
            self.search_space,
            tuning.guidance_objectives(),
            tuning.all_constraints(),
            rng=self.rng,
            exploration_probability=tuning.batch_exploration,
            diversity_weight=tuning.diversity_weight,
            uncertainty_weight=tuning.exploration_weight,
        )

        self.history = HistoryStore(
            self.artifact_dir / "history.jsonl",
            resume=tuning.resume,
        )
        self.resumed_evaluations = self.history.evaluation_count
        self.manifest_path = self.artifact_dir / "run_manifest.json"
        self.checkpoint_path = self.artifact_dir / "checkpoint.json"
        self.round_dir = self.artifact_dir / "rounds"

        if self.history.evaluation_count > tuning.budget:
            raise TuningError(
                f"resumed history has {self.history.evaluation_count} evaluations, "
                f"exceeding configured budget {tuning.budget}"
            )
        self._validate_resumed_candidates()

    @classmethod
    def from_project(
        cls,
        project: LoadedProject,
        *,
        runner: BaseRunner,
        proposer: CandidateProposer | None = None,
        llm_client: CompletionClient | None = None,
        before_evaluation: Callable[[EvaluationRequest], None] | None = None,
        after_evaluation: Callable[[EvaluationRecord], None] | None = None,
    ) -> "Tuner":
        """Construct from :func:`eigen.config.load_project` output."""

        config = project.config
        settings = config.runner.settings
        evaluation_timeout = float(settings.get("timeout_s", settings.get("timeout", 86_400.0)))
        proposal_timeout = config.llm.timeout_s if config.llm is not None else 120.0
        return cls(
            profile=project.profile,
            tuning=config.tuning,
            execution=config.execution,
            runner=runner,
            artifact_dir=project.artifact_dir,
            experiment_name=config.experiment_name,
            proposer=proposer,
            llm_client=llm_client,
            before_evaluation=before_evaluation,
            after_evaluation=after_evaluation,
            evaluation_timeout_s=evaluation_timeout,
            proposal_timeout_s=proposal_timeout,
        )

    def _runtime_values(self) -> dict[str, Any]:
        available = {
            "experiment_name": self.experiment_name,
            "connection_params": dict(self.execution.get("connection_params", {})),
            "upload_parallel": self.execution.get("upload_parallel"),
            "search_parallel": self.execution.get("search_parallel"),
            "top_k": self.execution.get("top_k"),
            "batch_size": self.execution.get("batch_size"),
            "vector_size": self.execution.get("vector_size"),
        }
        declared = set(self.profile.experiment.runtime_bindings) | set(
            self.profile.experiment.runtime_context
        )
        return {
            name: value
            for name, value in available.items()
            if name in declared and value is not None
        }

    def _workload(self) -> WorkloadSpec:
        return WorkloadSpec(
            dataset=str(self.execution["dataset"]),
            distance=self.execution.get("distance"),
            vector_size=self.execution.get("vector_size"),
            top_k=int(self.execution.get("top_k", 10)),
            concurrency=int(self.execution.get("search_parallel", 1)),
            filtered=bool(self.execution.get("filtered", False)),
            sparse=bool(self.execution.get("sparse", False)),
        )

    def _default_proposer(self, llm_client: CompletionClient | None) -> CandidateProposer:
        if self.tuning.strategy in {"random", "knn"}:
            return self.random_proposer
        if llm_client is None:
            raise TuningError("CALM requires an LLM completion client")

        llm_proposer = OpenAICompatibleProposer(
            self.search_space,
            llm_client,
            runtime=self.runtime,
            objective_metric=self.tuning.objective_metric,
            constraint_metric=self.tuning.constraint_metric,
            constraint_threshold=self.tuning.recall_threshold,
            history_limit=self.tuning.history_limit,
            task=self.task,
            objectives=self.tuning.objectives,
            guidance_objectives=self.tuning.guidance_objectives(),
            constraints=self.tuning.all_constraints(),
        )
        return HybridProposer(llm_proposer, self.random_proposer)

    def _validate_resumed_candidates(self) -> None:
        for record in self.history.records:
            canonical = self.search_space.canonicalize(
                record.candidate,
                runtime=self.runtime,
            )
            if candidate_key(canonical) != record.candidate_key:
                raise TuningError(
                    "resumed history is incompatible with the selected engine profile"
                )

    def _resume_contract(self) -> dict[str, Any]:
        tuning_policy = self.tuning.model_dump(mode="json")
        # Increasing a completed experiment's budget is a supported resume
        # operation; changing its proposal/selection policy is not silent.
        tuning_policy.pop("budget", None)
        tuning_policy.pop("resume", None)
        tuning_policy.pop("transfer_candidates_per_region", None)
        runner_settings = dict(getattr(self.runner.context, "settings", {}))
        return {
            "optimizer_contract_version": 3,
            "initialization_policy": (
                "explicit_override"
                if self.tuning.initial_samples is not None
                else "one_per_region"
                if self.tuning.strategy == "calm"
                else "random_design_14"
            ),
            "guidance_objectives": [o.model_dump() for o in self.tuning.guidance_objectives()],
            "profile_fingerprint": profile_fingerprint(self.profile),
            "experiment_name": self.experiment_name,
            "engine": self.profile.adapter.engine,
            "objective_metric": self.tuning.objective_metric,
            "constraint_metric": self.tuning.constraint_metric,
            "constraint_threshold": self.tuning.recall_threshold,
            "seed": self.tuning.seed,
            "runner": self.runner.manifest(),
            "runner_settings_fingerprint": fingerprint(runner_settings, length=64),
            "execution_fingerprint": fingerprint(self.execution, length=64),
            "tuning_policy_fingerprint": fingerprint(tuning_policy, length=64),
            "llm_fingerprint": fingerprint(self.llm_contract, length=64),
            "evaluation_timeout_s": self.evaluation_timeout_s,
            # Runtime values may contain connection credentials. Persist only a
            # one-way fingerprint, never the values themselves.
            "runtime_fingerprint": fingerprint(self.runtime, length=64),
            "workload": {
                "dataset": self.workload.dataset,
                "distance": self.workload.distance,
                "vector_size": self.workload.vector_size,
                "top_k": self.workload.top_k,
                "concurrency": self.workload.concurrency,
                "filtered": self.workload.filtered,
                "sparse": self.workload.sparse,
            },
        }

    def _ensure_manifest(self) -> None:
        contract = self._resume_contract()
        contract_hash = fingerprint(contract, length=64)
        if self.tuning.resume and self.manifest_path.exists():
            try:
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise TuningError(f"cannot read resume manifest: {error}") from error
            if existing.get("resume_contract_hash") != contract_hash:
                raise TuningError(
                    "artifact directory belongs to an incompatible profile/workload run"
                )
            return
        if self.tuning.resume and self.history.evaluation_count and not self.manifest_path.exists():
            raise TuningError("cannot resume history without its run_manifest.json")

        atomic_write_json(
            self.manifest_path,
            {
                "schema_version": 1,
                "created_at": _utc_now(),
                "resume_contract": contract,
                "resume_contract_hash": contract_hash,
                "profile_id": self.profile.id,
                "runner": self.runner.manifest(),
                "tuning": self.tuning.model_dump(mode="json"),
                "initial_evaluations": self._initial_design_size(),
            },
        )

    @property
    def remaining_budget(self) -> int:
        return self.tuning.budget - self.history.evaluation_count

    def run(self) -> TuningResult:
        """Execute until the exact evaluation budget is consumed or fail safely."""

        try:
            self._ensure_manifest()
            self._write_checkpoint(round_index=None)
            self._run_initial_design()

            previous_rounds = [
                int(path.stem.removeprefix("round-"))
                for path in self.round_dir.glob("round-*.json")
                if path.stem.removeprefix("round-").isdigit()
            ]
            round_index = max(previous_rounds, default=-1) + 1
            while self.remaining_budget > 0:
                # Every decision has a seed derived from durable evaluation count.
                # This avoids restarting the random stream at seed 0 on resume.
                self.rng.seed(f"{self.tuning.seed}:{self.history.evaluation_count}")
                pool, selected_regions = self._proposal_pool()
                if not pool:
                    raise TuningError(
                        "candidate proposer could not produce an unseen valid candidate; "
                        f"{self.remaining_budget} budget units remain"
                    )

                context_records = self.history.successful(
                    required_metrics=(
                        self.tuning.objective_metric,
                        self.tuning.constraint_metric,
                    ),
                    limit=self.tuning.history_limit,
                )
                evaluation_count = min(
                    self.tuning.evaluations_per_round,
                    self.remaining_budget,
                    len(pool),
                )
                selected = []
                if self.tuning.strategy == "calm":
                    self.surrogate.fit(context_records)
                    predictions = self.surrogate.predict_many(
                        pool, deadline=time.monotonic() + self.proposal_timeout_s
                    )
                    selected = self.calm_acquisition.select(
                        predictions, self.history.records, evaluation_count
                    )
                elif self.tuning.strategy == "knn":
                    self.surrogate.fit(context_records)
                    predictions = self.surrogate.predict_many(pool)
                    selected = self.acquisition.select(
                        predictions, self.history.records, evaluation_count
                    )
                evaluated_sequences: list[int] = []
                if self.tuning.strategy == "random":
                    # Plain random is not surrogate-screened; LHS is a separate baseline.
                    for candidate in pool[:evaluation_count]:
                        record = self._evaluate(
                            candidate, region=self.partitioner.region_for(candidate)
                        )
                        evaluated_sequences.append(record.sequence)
                for acquisition_score in selected:
                    candidate = acquisition_score.prediction.candidate
                    region = self.partitioner.region_for(candidate)
                    record = self._evaluate(
                        candidate,
                        region=region,
                        acquisition_score=acquisition_score,
                    )
                    evaluated_sequences.append(record.sequence)

                self._write_round_artifact(
                    round_index,
                    pool,
                    selected_regions,
                    selected,
                    evaluated_sequences,
                )
                self._write_checkpoint(round_index=round_index)
                round_index += 1

            return self.result()
        finally:
            self.runner.close()

    def _initial_design_size(self) -> int:
        requested = self.tuning.initial_samples
        if requested is None:
            requested = len(self.partitioner.regions) if self.tuning.strategy == "calm" else 14
        if self.tuning.strategy == "calm":
            requested = max(requested, len(self.partitioner.regions))
        return min(requested, self.tuning.budget)

    def _run_initial_design(self) -> None:
        target = self._initial_design_size()
        regions = self.partitioner.regions
        cursor = self.history.evaluation_count
        consecutive_exhausted = 0
        while self.history.evaluation_count < target:
            region = regions[cursor % len(regions)]
            cursor += 1
            candidates = self.random_proposer.propose(
                1,
                region=region,
                history=self.history.records,
                excluded_keys=self.history.seen_keys,
            )
            if not candidates:
                consecutive_exhausted += 1
                if consecutive_exhausted >= len(regions):
                    raise TuningError("random initial design exhausted the canonical search space")
                continue
            consecutive_exhausted = 0
            self._evaluate(candidates[0], region=region)

    def _proposal_pool(self) -> tuple[list[dict[str, Any]], tuple[Region, ...]]:
        probes = {}
        if self.tuning.strategy == "calm":
            probe_points = []
            probe_keys = set()
            for region in self.partitioner.regions:
                points = self.random_proposer.propose(
                    self.tuning.region_probe_count, region=region, excluded_keys=probe_keys
                )
                probe_points.extend(points)
                probe_keys.update(candidate_key(c) for c in points)
            self.surrogate.fit(self.history.records)
            for prediction in self.surrogate.predict_many(
                probe_points, deadline=time.monotonic() + self.proposal_timeout_s
            ):
                region = self.partitioner.region_for(prediction.candidate)
                probes.setdefault(region.id, []).append(prediction)
        selected_regions = self.partitioner.select_regions(
            self.history.records,
            self.tuning.regions_per_round,
            objective_metric=self.tuning.objective_metric,
            constraint_metric=self.tuning.constraint_metric,
            threshold=self.tuning.recall_threshold,
            exploration_weight=self.tuning.exploration_weight,
            objectives=self.tuning.guidance_objectives(),
            constraints=self.tuning.all_constraints(),
            probes=probes,
            min_observations=self.tuning.region_min_observations,
            rng=self.rng,
            exploration_probability=(
                1.0 if self.tuning.strategy == "random" else self.tuning.region_exploration
            ),
        )
        target = self.tuning.proposals_per_round
        base, extra = divmod(target, len(selected_regions))
        pool: list[dict[str, Any]] = []
        pool_keys: set[str] = set()
        deadline = time.monotonic() + self.proposal_timeout_s

        for index, region in enumerate(selected_regions):
            allocation = base + (1 if index < extra else 0)
            excluded = set(self.history.seen_keys) | pool_keys
            random_count = (
                sum(self.rng.random() < self.tuning.proposal_exploration for _ in range(allocation))
                if self.tuning.strategy == "calm"
                else 0
            )
            proposed = self.random_proposer.propose(
                random_count, region=region, excluded_keys=excluded, deadline=deadline
            )
            proposed += self.proposer.propose(
                allocation - len(proposed),
                region=region,
                history=self.history.records,
                excluded_keys=excluded | {candidate_key(c) for c in proposed},
                deadline=deadline,
            )
            for candidate in proposed:
                canonical = self.search_space.canonicalize(
                    candidate,
                    runtime=self.runtime,
                    reject_inactive=False,
                )
                key = candidate_key(canonical)
                if key in self.history.seen_keys or key in pool_keys:
                    continue
                if not region.matches(canonical):
                    continue
                pool_keys.add(key)
                pool.append(canonical)

        # Validated random proposals fill short batches and exhausted regions.
        considered_regions = list(selected_regions)
        if len(pool) < target:
            fallback_deadline = time.monotonic() + RANDOM_FALLBACK_TIMEOUT_S
            # A conditional branch can be finite (FLAT/exact is often a
            # singleton).  Once exhausted, continue through the remaining
            # ranked regions instead of terminating an otherwise valid run.
            all_ranked = self.partitioner.select_regions(
                self.history.records,
                len(self.partitioner.regions),
                objective_metric=self.tuning.objective_metric,
                constraint_metric=self.tuning.constraint_metric,
                threshold=self.tuning.recall_threshold,
                exploration_weight=self.tuning.exploration_weight,
                objectives=self.tuning.guidance_objectives(),
                constraints=self.tuning.all_constraints(),
                rng=self.rng,
                exploration_probability=self.tuning.region_exploration,
            )
            fallback_regions: list[Region] = []
            fallback_region_ids: set[str] = set()
            for region in (*selected_regions, *all_ranked):
                if region.id in fallback_region_ids:
                    continue
                fallback_region_ids.add(region.id)
                fallback_regions.append(region)
            for region in fallback_regions:
                fallback = self.random_proposer.propose(
                    target - len(pool),
                    region=region,
                    history=self.history.records,
                    excluded_keys=set(self.history.seen_keys) | pool_keys,
                    deadline=fallback_deadline,
                )
                for candidate in fallback:
                    key = candidate_key(candidate)
                    if key not in pool_keys and key not in self.history.seen_keys:
                        pool_keys.add(key)
                        pool.append(candidate)
                        if region not in considered_regions:
                            considered_regions.append(region)
                if len(pool) >= target:
                    break
        return pool[:target], tuple(considered_regions)

    def _evaluate(
        self,
        candidate: Mapping[str, Any],
        *,
        region: Region,
        acquisition_score: AcquisitionScore | CALMScore | None = None,
    ) -> EvaluationRecord:
        if self.remaining_budget <= 0:
            raise TuningError("evaluation budget is exhausted")
        canonical = self.search_space.canonicalize(candidate, runtime=self.runtime)
        if self.history.contains(canonical):
            raise TuningError("refusing to evaluate a duplicate canonical candidate")

        sequence = self.history.evaluation_count
        run_id = safe_name(
            f"{self.experiment_name}-e{sequence:04d}-{candidate_key(canonical)[:12]}"
        )
        render_runtime = dict(self.runtime)
        if "experiment_name" in self.profile.experiment.runtime_bindings:
            render_runtime["experiment_name"] = run_id
        rendered = self.renderer.render(canonical, runtime=render_runtime)
        request = EvaluationRequest(
            run_id=run_id,
            engine_id=self.profile.adapter.engine,
            candidate=canonical,
            rendered_experiment=rendered,
            workload=self.workload,
            seed=self.tuning.seed + sequence,
            timeout_s=self.evaluation_timeout_s,
        )
        started_at = _utc_now()
        # Lifecycle/control-plane failures happen before a database evaluation
        # is invoked.  They must abort the study without consuming budget or
        # being mislabeled as an ordinary failed configuration.
        if self.before_evaluation is not None:
            self.before_evaluation(request)

        interrupted: BaseException | None = None
        try:
            observation = self.runner.evaluate(request)
            if not isinstance(observation, Observation):
                raise TypeError("runner.evaluate must return an Observation")
            observation = self._validated_observation(observation)
        except BaseException as error:  # record every invoked evaluation, even Ctrl-C
            observation = Observation(
                status=RunStatus.FAILED,
                error=f"{type(error).__name__}: {error}",
            )
            if not isinstance(error, Exception):
                interrupted = error

        prediction = acquisition_score.prediction.to_dict() if acquisition_score is not None else {}
        acquisition = acquisition_score.to_dict() if acquisition_score is not None else {}
        record = EvaluationRecord.from_observation(
            sequence=sequence,
            run_id=run_id,
            candidate=canonical,
            observation=observation,
            region_id=region.id,
            prediction=prediction,
            acquisition=acquisition,
            started_at=started_at,
        )
        self.history.append(record)
        atomic_write_json(self.artifact_dir / "pareto_archive.json", self._archive_payload())
        self._write_checkpoint(round_index=None)
        if self.after_evaluation is not None:
            self.after_evaluation(record)
        if interrupted is not None:
            raise interrupted
        return record

    def _validated_observation(self, observation: Observation) -> Observation:
        if not observation.ok:
            return observation
        required = {self.tuning.objective_metric, self.tuning.constraint_metric}
        required.update(c.metric for c in self.tuning.all_constraints())
        required.update(o.metric for o in self.tuning.objectives)
        try:
            raw_values = {name: observation.metrics[name] for name in required}
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in observation.metrics.values()
            ):
                raise TypeError("metrics must be finite JSON numbers")
            values = {name: float(value) for name, value in raw_values.items()}
        except (KeyError, TypeError, ValueError) as error:
            return Observation(
                status=RunStatus.FAILED,
                auxiliary=dict(observation.auxiliary),
                artifacts=list(observation.artifacts),
                error=f"runner result lacks finite required metrics: {error}",
            )
        if not all(math.isfinite(value) for value in values.values()):
            return Observation(
                status=RunStatus.FAILED,
                auxiliary=dict(observation.auxiliary),
                artifacts=list(observation.artifacts),
                error="runner result contains non-finite required metrics",
            )
        if "qps" in values and values["qps"] <= 0:
            return Observation(
                status=RunStatus.FAILED,
                auxiliary=dict(observation.auxiliary),
                artifacts=list(observation.artifacts),
                error="runner result qps must be positive",
            )
        if "recall" in values and not 0.0 <= values["recall"] <= 1.0:
            return Observation(
                status=RunStatus.FAILED,
                auxiliary=dict(observation.auxiliary),
                artifacts=list(observation.artifacts),
                error="runner result recall must be in [0, 1]",
            )
        if any(
            values.get(name, 0) < 0
            for name in ("build_total_time_s", "mean_latency_s", "memory_bytes")
        ):
            return Observation(status=RunStatus.FAILED, error="negative efficiency metric")
        return observation

    def _write_round_artifact(
        self,
        round_index: int,
        pool: Sequence[Mapping[str, Any]],
        regions: Sequence[Region],
        selected: Sequence[AcquisitionScore | CALMScore],
        evaluated_sequences: Sequence[int],
    ) -> None:
        atomic_write_json(
            self.round_dir / f"round-{round_index:04d}.json",
            {
                "round": round_index,
                "created_at": _utc_now(),
                "regions": [region.id for region in regions],
                "proposal_keys": [candidate_key(candidate) for candidate in pool],
                "ranked": [
                    {
                        "candidate_key": candidate_key(item.prediction.candidate),
                        **item.prediction.to_dict(),
                        **item.to_dict(),
                    }
                    for item in selected
                ],
                "evaluated_sequences": list(evaluated_sequences),
            },
        )

    def _write_checkpoint(self, *, round_index: int | None) -> None:
        best = self._best_feasible()
        atomic_write_json(
            self.checkpoint_path,
            {
                "schema_version": 1,
                "updated_at": _utc_now(),
                "round": round_index,
                "budget": self.tuning.budget,
                "completed_evaluations": self.history.evaluation_count,
                "remaining_budget": self.remaining_budget,
                "best_feasible": (
                    {
                        "sequence": best.sequence,
                        "candidate": best.candidate,
                        "metrics": best.metrics,
                    }
                    if best is not None
                    else None
                ),
                "history_path": str(self.history.path),
            },
        )

    def result(self) -> TuningResult:
        best = self._best_feasible()
        return TuningResult(
            history_path=self.history.path,
            checkpoint_path=self.checkpoint_path,
            budget=self.tuning.budget,
            completed_evaluations=self.history.evaluation_count,
            best_candidate=dict(best.candidate) if best is not None else None,
            best_metrics=dict(best.metrics) if best is not None else None,
            pareto_candidates=self._archive_payload(),
            # Compatibility name for the same complete feasible archive.
            transfer_candidates=self._archive_payload(),
            resumed_evaluations=self.resumed_evaluations,
            llm_wall_s=sum(client.elapsed_s for client in self.logged_clients),
        )

    def _best_feasible(self) -> EvaluationRecord | None:
        return max(
            (
                r
                for r in self.history.records
                if r.ok
                and self.tuning.objective_metric in r.metrics
                and feasible(r.metrics, self.tuning.all_constraints())
            ),
            key=lambda r: r.metrics[self.tuning.objective_metric],
            default=None,
        )

    def _archive_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "candidate": r.candidate,
                "metrics": r.metrics,
                "sequence": r.sequence,
                "region_id": r.region_id,
            }
            for r in transfer_pool(self.history.records, self.tuning)
        ]


__all__ = ["Tuner", "TuningError", "TuningResult"]
