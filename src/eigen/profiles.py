"""Loading and workload validation for declarative engine profiles."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

from pydantic import ValidationError

from eigen.api import WorkloadSpec
from eigen.errors import ConfigurationError
from eigen.models import EngineProfile

_PROFILE_PACKAGE = "eigen.resources.profiles"
_DISTANCE_ALIASES = {
    "l2": "l2",
    "euclidean": "l2",
    "cosine": "cosine",
    "angular": "cosine",
    "dot": "dot",
    "ip": "dot",
    "inner_product": "dot",
    "inner-product": "dot",
}


def list_builtin_profiles() -> list[str]:
    """Return stable IDs for all packaged JSON profiles."""

    root = resources.files(_PROFILE_PACKAGE)
    return sorted(
        item.name.removesuffix(".json") for item in root.iterdir() if item.name.endswith(".json")
    )


def load_profile(source: str | Path) -> EngineProfile:
    """Load a built-in profile ID or an explicit JSON profile path.

    Bare names such as ``milvus-hnsw-dense`` resolve against packaged
    resources. Strings ending in ``.json`` and :class:`Path` objects are
    interpreted as filesystem paths.
    """

    label: str
    try:
        if isinstance(source, Path) or str(source).endswith(".json"):
            path = Path(source).expanduser()
            label = str(path)
            payload = path.read_text(encoding="utf-8")
        else:
            profile_id = str(source)
            label = f"built-in profile {profile_id!r}"
            item = resources.files(_PROFILE_PACKAGE).joinpath(f"{profile_id}.json")
            if not item.is_file():
                available = ", ".join(list_builtin_profiles())
                raise ConfigurationError(
                    f"unknown built-in profile {profile_id!r}; available: {available}"
                )
            payload = item.read_text(encoding="utf-8")
    except ConfigurationError:
        raise
    except OSError as error:
        raise ConfigurationError(f"cannot read engine profile {source!s}: {error}") from error

    try:
        profile = EngineProfile.model_validate_json(payload)
    except ValidationError as error:
        raise ConfigurationError(f"invalid {label}: {error}") from error

    if not isinstance(source, Path) and not str(source).endswith(".json"):
        if profile.id != str(source):
            raise ConfigurationError(
                f"built-in filename {source!r} does not match profile id {profile.id!r}"
            )
    return profile


def normalize_distance(distance: str) -> str:
    """Normalize benchmark distance aliases to the profile vocabulary."""

    normalized = _DISTANCE_ALIASES.get(distance.strip().lower())
    if normalized is None:
        raise ConfigurationError(f"unknown vector distance {distance!r}")
    return normalized


def validate_workload(profile: EngineProfile, workload: WorkloadSpec) -> None:
    """Reject a workload that the selected profile cannot faithfully run."""

    if not workload.dataset.strip():
        raise ConfigurationError("workload dataset cannot be empty")
    if workload.top_k <= 0:
        raise ConfigurationError("workload top_k must be positive")
    if workload.concurrency <= 0:
        raise ConfigurationError("workload concurrency must be positive")
    if workload.vector_size is not None and workload.vector_size <= 0:
        raise ConfigurationError("workload vector_size must be positive")

    vector_kind = "sparse" if workload.sparse else "dense"
    if vector_kind not in profile.capabilities.vector_kinds:
        raise ConfigurationError(f"profile {profile.id!r} does not support {vector_kind} workloads")

    if workload.distance is not None:
        distance = normalize_distance(workload.distance)
        if distance not in profile.capabilities.distances:
            raise ConfigurationError(
                f"profile {profile.id!r} does not support distance {distance!r}"
            )

    if workload.filtered and not profile.capabilities.filters:
        raise ConfigurationError(
            f"profile {profile.id!r} cannot faithfully execute filtered workloads"
        )


def profile_fingerprint(profile: EngineProfile) -> str:
    """Return a stable SHA-256 of the fully validated profile."""

    payload = json.dumps(
        profile.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "list_builtin_profiles",
    "load_profile",
    "normalize_distance",
    "profile_fingerprint",
    "validate_workload",
]
