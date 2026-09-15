"""Measured-history-conditioned LLM surrogate for CALM (paper Section 5)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mutune.config import MetricConstraint, ObjectiveSpec
from mutune.llm import CompletionClient, CompletionResult, LLMError, parse_json_response

from .history import EvaluationRecord, candidate_key
from .proposer import _search_space_description
from .surrogate import MetricPrediction


@dataclass(frozen=True)
class LLMPrediction:
    candidate: dict[str, Any]
    metrics: dict[str, MetricPrediction]
    probability_feasible: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability_feasible": self.probability_feasible,
            "metrics": {
                name: {"mean": p.mean, "stddev": p.stddev} for name, p in self.metrics.items()
            },
        }


class LLMSurrogate:
    """Predict requested metrics and *joint* feasibility, with strict ID matching.

    Probabilities/uncertainties are LLM estimates, not calibrated confidence
    guarantees. Persistent randomized evaluation does not depend on accuracy.
    Missing/malformed predictions retry and then fail; never silently use KNN.
    """

    def __init__(
        self,
        search_space: Any,
        client: CompletionClient,
        *,
        objectives: Sequence[ObjectiveSpec],
        constraints: Sequence[MetricConstraint],
        task: Mapping[str, Any],
        history_limit: int = 100,
        batch_size: int = 24,
        max_attempts: int = 3,
    ) -> None:
        self.search_space, self.client = search_space, client
        self.objectives, self.constraints = list(objectives), list(constraints)
        self.task, self.history_limit = dict(task), history_limit
        self.batch_size, self.max_attempts = batch_size, max_attempts
        self.records: list[EvaluationRecord] = []
        self.metric_names = sorted({o.metric for o in objectives} | {c.metric for c in constraints})

    def fit(self, records: Sequence[EvaluationRecord]) -> "LLMSurrogate":
        self.records = list(records)[-self.history_limit :]
        return self

    def predict_many(
        self, candidates: Sequence[Mapping[str, Any]], *, deadline: float | None = None
    ) -> list[LLMPrediction]:
        predictions = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            keyed = {candidate_key(c): dict(c) for c in batch}
            if len(keyed) != len(batch):
                raise ValueError("surrogate candidates must be unique")
            contract = {
                "task": "Predict vector-database measurements; do not propose new configurations",
                "workload": self.task,
                "search_space": _search_space_description(self.search_space),
                "objectives": [o.model_dump() for o in self.objectives],
                "constraints": [c.model_dump() for c in self.constraints],
                "history": [
                    {"candidate": r.candidate, "metrics": r.metrics, "status": r.status}
                    for r in self.records
                ],
                "candidates": [{"id": key, "candidate": value} for key, value in keyed.items()],
                "output_contract": {
                    "predictions": [
                        {
                            "id": "copy candidate id exactly",
                            "probability_feasible": "number in [0,1]: probability ALL constraints hold",
                            "metrics": {
                                name: {
                                    "mean": "finite numeric predicted value",
                                    "stddev": "finite positive uncertainty in metric units",
                                }
                                for name in self.metric_names
                            },
                        }
                    ]
                },
            }
            error = None
            for _ in range(self.max_attempts):
                try:
                    response = self.client.complete(
                        json.dumps(contract, ensure_ascii=False), deadline=deadline
                    )
                    payload = parse_json_response(
                        response.content
                        if isinstance(response, CompletionResult)
                        else str(response)
                    )
                    parsed = self._parse(payload, keyed)
                    predictions.extend(parsed)
                    break
                except (LLMError, ValueError, TypeError, KeyError) as exc:
                    error = exc
            else:
                raise LLMError(f"LLM surrogate failed its prediction contract: {error}") from error
        return predictions

    def _parse(self, payload: Any, keyed: Mapping[str, dict[str, Any]]) -> list[LLMPrediction]:
        if not isinstance(payload, dict) or not isinstance(payload.get("predictions"), list):
            raise ValueError("response must contain a predictions array")
        found: dict[str, LLMPrediction] = {}
        for item in payload["predictions"]:
            if not isinstance(item, dict):
                raise ValueError("each prediction must be an object")
            key = item.get("id")
            if not isinstance(key, str) or key not in keyed or key in found:
                raise ValueError("missing, unknown, or duplicated prediction id")
            probability = _number(item["probability_feasible"])
            if not 0 <= probability <= 1:
                raise ValueError("feasibility probability must be in [0,1]")
            raw = item.get("metrics")
            if not isinstance(raw, dict) or set(raw) != set(self.metric_names):
                raise ValueError("prediction must provide exactly the requested metrics")
            metrics = {}
            for name in self.metric_names:
                value = raw[name]
                if not isinstance(value, dict):
                    raise ValueError("each metric needs mean and stddev")
                mean, stddev = _number(value["mean"]), _number(value["stddev"])
                if stddev <= 0 or (name == "recall" and not 0 <= mean <= 1):
                    raise ValueError("invalid metric uncertainty or recall mean")
                if (
                    name in {"qps", "mean_latency", "build_total_time_s", "memory_bytes"}
                    and mean < 0
                ):
                    raise ValueError("performance metrics must be nonnegative")
                metrics[name] = MetricPrediction(mean, stddev)
            found[key] = LLMPrediction(keyed[key], metrics, probability)
        if set(found) != set(keyed):
            raise ValueError("surrogate omitted candidates")
        return [found[key] for key in keyed]


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("predictions must be finite JSON numbers")
    return float(value)
