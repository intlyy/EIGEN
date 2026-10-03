"""Append-only, resumable optimization history."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from eigen.api import Observation, RunStatus
from eigen.utils import atomic_write_text, canonical_json, fingerprint


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached, deterministically serializable candidate."""

    encoded = canonical_json(dict(candidate))
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # defensive: Mapping should always encode as object
        raise TypeError("candidate must serialize to a JSON object")
    return decoded


def candidate_key(candidate: Mapping[str, Any]) -> str:
    return fingerprint(canonical_candidate(candidate), length=64)


@dataclass(slots=True)
class EvaluationRecord:
    """One runner invocation; every record consumes exactly one budget unit."""

    sequence: int
    run_id: str
    candidate: dict[str, Any]
    candidate_key: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    region_id: str | None = None
    prediction: dict[str, Any] = field(default_factory=dict)
    acquisition: dict[str, Any] = field(default_factory=dict)
    auxiliary: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: str = field(default_factory=_utc_now)
    finished_at: str = field(default_factory=_utc_now)

    @classmethod
    def from_observation(
        cls,
        *,
        sequence: int,
        run_id: str,
        candidate: Mapping[str, Any],
        observation: Observation,
        region_id: str | None = None,
        prediction: Mapping[str, Any] | None = None,
        acquisition: Mapping[str, Any] | None = None,
        started_at: str | None = None,
    ) -> "EvaluationRecord":
        normalized = canonical_candidate(candidate)
        metrics = {
            str(name): float(value)
            for name, value in observation.metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        }
        return cls(
            sequence=sequence,
            run_id=run_id,
            candidate=normalized,
            candidate_key=candidate_key(normalized),
            status=observation.status.value,
            metrics=metrics,
            region_id=region_id,
            prediction=dict(prediction or {}),
            acquisition=dict(acquisition or {}),
            auxiliary=dict(observation.auxiliary),
            artifacts=list(observation.artifacts),
            error=observation.error,
            started_at=started_at or _utc_now(),
            finished_at=_utc_now(),
        )

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.OK.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationRecord":
        candidate = canonical_candidate(payload["candidate"])
        expected_key = candidate_key(candidate)
        stored_key = str(payload.get("candidate_key", expected_key))
        if stored_key != expected_key:
            raise ValueError("history candidate_key does not match canonical candidate")
        status = str(payload["status"])
        if status not in {member.value for member in RunStatus}:
            raise ValueError(f"unknown history status: {status}")
        return cls(
            sequence=int(payload["sequence"]),
            run_id=str(payload["run_id"]),
            candidate=candidate,
            candidate_key=stored_key,
            status=status,
            metrics={str(k): float(v) for k, v in dict(payload.get("metrics", {})).items()},
            region_id=(str(payload["region_id"]) if payload.get("region_id") is not None else None),
            prediction=dict(payload.get("prediction", {})),
            acquisition=dict(payload.get("acquisition", {})),
            auxiliary=dict(payload.get("auxiliary", {})),
            artifacts=[str(value) for value in payload.get("artifacts", [])],
            error=str(payload["error"]) if payload.get("error") is not None else None,
            started_at=str(payload.get("started_at", "")),
            finished_at=str(payload.get("finished_at", "")),
        )


class HistoryStore:
    """JSONL store that can recover after an interrupted final append."""

    def __init__(self, path: str | Path, *, resume: bool = True) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            atomic_write_text(self.path, "")
        self._records = self._load() if self.path.exists() else []
        self._validate_sequences()
        self._keys = {record.candidate_key for record in self._records}

    def _load(self) -> list[EvaluationRecord]:
        raw = self.path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        records: list[EvaluationRecord] = []
        recovered_truncated_tail = False
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("JSONL entries must be objects")
                records.append(EvaluationRecord.from_dict(payload))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                # A process may die between write() and the terminating newline.  Only
                # an invalid final line is safely recoverable; earlier corruption is not.
                if index == len(lines) - 1 and raw and not raw.endswith("\n"):
                    recovered_truncated_tail = True
                    break
                raise ValueError(f"Invalid history entry on line {index + 1}: {error}") from error
        if recovered_truncated_tail:
            # Merely ignoring the tail is insufficient: a subsequent append would
            # continue on the same corrupt physical line.  Atomically replace it
            # with the validated prefix before the store accepts new records.
            clean_prefix = "".join(
                json.dumps(
                    record.to_dict(),
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
                for record in records
            )
            atomic_write_text(self.path, clean_prefix)
        return records

    def _validate_sequences(self) -> None:
        for expected, record in enumerate(self._records):
            if record.sequence != expected:
                raise ValueError(
                    f"History sequence is not contiguous: expected {expected}, "
                    f"found {record.sequence}"
                )

    @property
    def records(self) -> tuple[EvaluationRecord, ...]:
        return tuple(self._records)

    @property
    def evaluation_count(self) -> int:
        return len(self._records)

    @property
    def seen_keys(self) -> frozenset[str]:
        return frozenset(self._keys)

    def contains(self, candidate: Mapping[str, Any]) -> bool:
        return candidate_key(candidate) in self._keys

    def append(self, record: EvaluationRecord) -> None:
        if record.sequence != len(self._records):
            raise ValueError(f"record sequence must be {len(self._records)}, got {record.sequence}")
        if record.candidate_key in self._keys:
            raise ValueError("candidate has already been evaluated")
        line = json.dumps(record.to_dict(), separators=(",", ":"), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records.append(record)
        self._keys.add(record.candidate_key)

    def successful(
        self,
        *,
        required_metrics: Iterable[str] = (),
        limit: int | None = None,
    ) -> list[EvaluationRecord]:
        required = set(required_metrics)
        records = [
            record
            for record in self._records
            if record.ok
            and required.issubset(record.metrics)
            and all(math.isfinite(record.metrics[name]) for name in required)
        ]
        return records[-limit:] if limit is not None else records

    def best_feasible(
        self,
        *,
        objective_metric: str,
        constraint_metric: str,
        threshold: float,
    ) -> EvaluationRecord | None:
        feasible = [
            record
            for record in self.successful(required_metrics=(objective_metric, constraint_metric))
            if record.metrics[constraint_metric] >= threshold
        ]
        return max(feasible, key=lambda record: record.metrics[objective_metric], default=None)


__all__ = [
    "EvaluationRecord",
    "HistoryStore",
    "candidate_key",
    "canonical_candidate",
]
