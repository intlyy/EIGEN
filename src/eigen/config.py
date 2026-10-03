"""Project configuration loading and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from eigen.errors import ConfigurationError


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ExecutionConfig(StrictModel):
    """Engine-neutral workload and benchmark execution settings."""

    dataset: str
    host: str = "localhost"
    distance: Literal["l2", "cosine", "dot"] | None = None
    vector_size: int | None = Field(default=None, gt=0)
    top_k: int = Field(default=10, gt=0)
    upload_parallel: int | None = Field(default=None, gt=0)
    search_parallel: int = Field(default=1, gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    filtered: bool = False
    sparse: bool = False
    connection_params: dict[str, Any] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)
    dataset_description: str = ""


class RunnerConfig(StrictModel):
    plugin: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class LifecycleConfig(StrictModel):
    mode: Literal["external", "docker_compose"] = "external"
    settings: dict[str, Any] = Field(default_factory=dict)
    restart_between_evaluations: bool = False


class ObjectiveSpec(StrictModel):
    metric: str = Field(default="qps", min_length=1)
    direction: Literal["maximize", "minimize"] = "maximize"


class MetricConstraint(StrictModel):
    metric: str = Field(min_length=1)
    operator: Literal["ge", "le"] = "ge"
    threshold: float

    def satisfied(self, metrics: dict[str, float]) -> bool:
        value = metrics.get(self.metric)
        if value is None:
            return False
        return value >= self.threshold if self.operator == "ge" else value <= self.threshold


class TuningConfig(StrictModel):
    """CALM defaults; alternative optimizers are explicitly named ablations."""

    budget: int = Field(default=20, gt=0)
    # CALM's paper default is one evaluation per conditional region. Explicit
    # values are an experimental override; random/KNN retain their 14-point design.
    initial_samples: int | None = Field(default=None, gt=0)
    proposals_per_round: int = Field(default=12, gt=0)
    evaluations_per_round: int = Field(default=4, gt=0)
    regions_per_round: int = Field(default=1, gt=0)
    recall_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    objective_metric: str = "qps"
    constraint_metric: str = "recall"
    strategy: Literal["calm", "random", "knn", "llm", "hybrid"] = "calm"
    # Paper Eq. (1): maximize QPS subject to recall, with no build-time objective.
    # Additional efficiency objectives are extensions; guidance adds recall separately.
    objectives: list[ObjectiveSpec] = Field(default_factory=lambda: [ObjectiveSpec()])
    constraints: list[MetricConstraint] = Field(default_factory=list)
    region_exploration: float = Field(default=0.1, gt=0, le=1)
    proposal_exploration: float = Field(default=0.1, gt=0, le=1)
    batch_exploration: float = Field(default=0.1, gt=0, le=1)
    diversity_weight: float = Field(default=0.2, ge=0)
    region_min_observations: int = Field(default=3, gt=0)
    region_probe_count: int = Field(default=2, gt=0)
    surrogate_batch_size: int = Field(default=24, gt=0)
    # Accepted for old config files only; the complete frontier is always transferred.
    transfer_candidates_per_region: int | None = Field(default=None, ge=2, deprecated=True)
    seed: int = 42
    exploration_weight: float = Field(default=0.20, ge=0.0)
    history_limit: int = Field(default=100, gt=0)
    resume: bool = True

    @model_validator(mode="after")
    def validate_budget(self) -> "TuningConfig":
        if self.initial_samples is not None and self.initial_samples > self.budget:
            raise ValueError("initial_samples cannot exceed budget")
        if self.evaluations_per_round > self.proposals_per_round:
            raise ValueError("evaluations_per_round cannot exceed proposals_per_round")
        if self.objective_metric == self.constraint_metric:
            raise ValueError("objective_metric and constraint_metric must differ")
        names = [objective.metric for objective in self.objectives]
        if not names or len(names) != len(set(names)):
            raise ValueError("objectives must be nonempty with unique metric names")
        if self.constraint_metric in names:
            raise ValueError("recall is a feasibility constraint, not an efficiency objective")
        if not any(
            o.metric == self.objective_metric and o.direction == "maximize" for o in self.objectives
        ):
            raise ValueError("objectives must include the primary maximized objective_metric")
        if self.strategy in {"llm", "hybrid"}:
            self.strategy = "calm"
        return self

    def all_constraints(self) -> list[MetricConstraint]:
        return [
            MetricConstraint(metric=self.constraint_metric, threshold=self.recall_threshold),
            *self.constraints,
        ]

    def guidance_objectives(self) -> list[ObjectiveSpec]:
        """Coordinates for CALM's archive and hypervolume, not final ranking.

        The paper retains feasible QPS/recall trade-offs while its final answer
        maximizes QPS subject to recall. Explicit efficiency objectives extend
        this guidance space; recall still remains a hard feasibility constraint.
        """
        return [*self.objectives, ObjectiveSpec(metric=self.constraint_metric)]


class LLMConfig(StrictModel):
    """One generic OpenAI-compatible endpoint; secrets remain in the environment."""

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=8000, gt=0)
    timeout_s: float = Field(default=120.0, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)

    @field_validator("api_key_env")
    @classmethod
    def validate_env_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env must be an environment-variable name")
        return value

    @field_validator("extra_body")
    @classmethod
    def protect_completion_contract(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = {"model", "messages", "temperature", "max_tokens"} & set(value)
        if reserved:
            raise ValueError(
                f"llm.extra_body cannot override core request fields: {sorted(reserved)}"
            )
        return value


class ProjectConfig(StrictModel):
    schema_version: Literal[1] = 1
    experiment_name: str
    engine_profile: str
    artifact_dir: str = "./artifacts"
    execution: ExecutionConfig
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    llm: LLMConfig | None = None

    @field_validator("experiment_name")
    @classmethod
    def validate_experiment_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value):
            raise ValueError(
                "experiment_name must be 1-100 portable characters: letters, digits, _, ., -"
            )
        return value

    @model_validator(mode="after")
    def require_llm_settings(self) -> "ProjectConfig":
        if self.tuning.strategy == "calm" and self.llm is None:
            raise ValueError("tuning.strategy=calm requires an llm section")
        return self


@dataclass(frozen=True, slots=True)
class LoadedProject:
    config: ProjectConfig
    profile: Any
    source_path: Path
    artifact_dir: Path


def _resolve_optional_path(
    settings: dict[str, Any],
    key: str,
    base_dir: Path,
    *,
    bare_command: bool = False,
) -> None:
    value = settings.get(key)
    if not isinstance(value, str) or not value:
        return
    if bare_command and "/" not in value and "\\" not in value:
        return
    path = Path(value).expanduser()
    if not path.is_absolute():
        settings[key] = str((base_dir / path).resolve())
    else:
        settings[key] = str(path.resolve())


def load_project(path: str | Path) -> LoadedProject:
    """Load a project JSON plus its built-in or file-backed engine profile."""

    from eigen.profiles import load_profile

    source_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        config = ProjectConfig.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ConfigurationError(f"Failed to load project config {source_path}: {error}") from error

    base_dir = source_path.parent
    profile_ref = config.engine_profile
    profile_path = Path(profile_ref).expanduser()
    if not profile_path.is_absolute() and (base_dir / profile_path).exists():
        profile_ref = str((base_dir / profile_path).resolve())

    settings = dict(config.runner.settings)
    for key in ("repo_path", "dataset_cache", "dataset_path"):
        _resolve_optional_path(settings, key, base_dir)
    _resolve_optional_path(
        settings,
        "python_executable",
        base_dir,
        bare_command=True,
    )
    config.runner.settings = settings

    artifact_path = Path(config.artifact_dir).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = base_dir / artifact_path
    artifact_path = artifact_path.resolve()

    try:
        profile = load_profile(profile_ref)
    except Exception as error:
        raise ConfigurationError(f"Failed to load engine profile {profile_ref}: {error}") from error

    return LoadedProject(
        config=config,
        profile=profile,
        source_path=source_path,
        artifact_dir=artifact_path,
    )
