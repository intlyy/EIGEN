"""Minimal OpenAI-compatible text completion support.

The core project deliberately does not depend on an LLM SDK.  This module uses
the standard library, reads credentials only from an environment variable, and
keeps retry/deadline handling explicit so a tuning run cannot wait forever.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from mutune.config import LLMConfig
from mutune.errors import MuTuneError


class LLMError(MuTuneError):
    """Raised when an LLM request or response cannot satisfy its contract."""


class LLMHTTPError(LLMError):
    """HTTP error retaining only the status needed for retry decisions."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status in {408, 409, 425, 429} or 500 <= self.status <= 599


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Normalized completion text and optional token accounting."""

    content: str
    usage: dict[str, int] = field(default_factory=dict)


class CompletionClient(Protocol):
    """Small interface used by the LLM candidate proposer."""

    def complete(self, prompt: str, *, deadline: float | None = None) -> CompletionResult:
        """Return one completion before an absolute monotonic deadline."""


class LoggedCompletionClient:
    """Record inference role, latency, usage and configured (not guessed) cost."""

    def __init__(self, client: CompletionClient, directory: Path, role: str) -> None:
        self.client, self.directory, self.role = client, directory, role

    def complete(self, prompt: str, *, deadline: float | None = None) -> CompletionResult:
        self.directory.mkdir(parents=True, exist_ok=True)
        call_id = uuid4().hex
        started = time.monotonic()
        record: dict[str, Any] = {
            "id": call_id,
            "role": self.role,
            "prompt_sha256": sha256(prompt.encode()).hexdigest(),
        }
        try:
            response = self.client.complete(prompt, deadline=deadline)
            if not isinstance(response, CompletionResult):
                response = CompletionResult(content=str(response))
            record.update(status="ok", usage=response.usage)
            config = getattr(self.client, "config", None)
            input_rate = getattr(config, "input_cost_per_million", None)
            output_rate = getattr(config, "output_cost_per_million", None)
            record["model"] = getattr(config, "model", None)
            record["cost_usd"] = (
                None
                if input_rate is None
                or output_rate is None
                or "prompt_tokens" not in response.usage
                or "completion_tokens" not in response.usage
                else (
                    response.usage["prompt_tokens"] * input_rate
                    + response.usage["completion_tokens"] * output_rate
                )
                / 1e6
            )
            (self.directory / f"{call_id}.json").write_text(
                json.dumps(
                    {"prompt": json.loads(prompt), "response": response.content},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return response
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__, cost_usd=None)
            raise
        finally:
            record["elapsed_s"] = time.monotonic() - started
            with (self.directory / "calls.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_s: float,
) -> Mapping[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - configured endpoint
            raw = response.read()
    except HTTPError as error:
        # Do not copy arbitrary server response bodies into logs or artifacts.
        raise LLMHTTPError(error.code, f"LLM endpoint returned HTTP {error.code}") from error
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LLMError("LLM endpoint returned a non-JSON response") from error
    if not isinstance(decoded, Mapping):
        raise LLMError("LLM endpoint returned a non-object JSON response")
    return decoded


class OpenAICompatibleClient:
    """Dependency-free client for an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        max_attempts: int = 3,
        retry_base_s: float = 0.25,
        transport: Transport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if retry_base_s < 0:
            raise ValueError("retry_base_s cannot be negative")
        self.config = config
        self.max_attempts = max_attempts
        self.retry_base_s = retry_base_s
        self._transport = transport or _default_transport
        self._monotonic = monotonic
        self._sleep = sleep

    @property
    def endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _credentials(self) -> str:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise LLMError(
                f"Required LLM credential environment variable is not set: "
                f"{self.config.api_key_env}"
            )
        return api_key

    def complete(self, prompt: str, *, deadline: float | None = None) -> CompletionResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        api_key = self._credentials()
        now = self._monotonic()
        effective_deadline = deadline if deadline is not None else now + self.config.timeout_s
        if not math.isfinite(effective_deadline) or effective_deadline <= now:
            raise LLMError("LLM request deadline has already expired")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": ("Return only valid JSON that follows the user's requested schema."),
                },
                {"role": "user", "content": prompt},
            ],
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        payload.update(self.config.extra_body)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "muTune/0.1",
        }

        last_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            remaining = effective_deadline - self._monotonic()
            if remaining <= 0:
                break
            timeout_s = min(self.config.timeout_s, remaining)
            try:
                response = self._transport(self.endpoint, headers, payload, timeout_s)
                return self._normalize_response(response)
            except LLMHTTPError as error:
                last_error = error
                if not error.retryable:
                    raise
            except (LLMError, URLError, TimeoutError, socket.timeout, OSError) as error:
                last_error = error

            if attempt + 1 >= self.max_attempts:
                break
            delay = self.retry_base_s * (2**attempt)
            remaining = effective_deadline - self._monotonic()
            if remaining <= 0:
                break
            self._sleep(min(delay, remaining))

        if effective_deadline <= self._monotonic():
            raise LLMError("LLM request deadline expired") from last_error
        raise LLMError(f"LLM request failed after {self.max_attempts} attempts") from last_error

    @staticmethod
    def _normalize_response(response: Mapping[str, Any]) -> CompletionResult:
        try:
            choices = response["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError("LLM response is missing choices[0].message.content") from error
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM response content is empty")

        raw_usage = response.get("usage", {})
        usage: dict[str, int] = {}
        if isinstance(raw_usage, Mapping):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = raw_usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    usage[key] = value
        return CompletionResult(content=content, usage=usage)


_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _extract_balanced_json(text: str) -> str:
    """Return the first balanced JSON object/array, respecting quoted strings."""

    start = next((index for index, char in enumerate(text) if char in "[{"), None)
    if start is None:
        return ""

    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "}]":
            if not stack or stack[-1] != pairs[char]:
                return ""
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return ""


def parse_json_response(text: str) -> Any:
    """Parse JSON from plain, fenced, or prose-wrapped LLM output.

    Unlike permissive legacy parsers, this function never rewrites quotes or
    attempts to interpret Python literals.  It therefore cannot silently alter
    candidate values.
    """

    if not isinstance(text, str) or not text.strip():
        raise LLMError("Cannot parse JSON from an empty LLM response")
    cleaned = _THINK_BLOCK.sub("", text).strip()

    candidates = [cleaned]
    candidates.extend(match.group(1).strip() for match in _FENCED_BLOCK.finditer(cleaned))
    balanced = _extract_balanced_json(cleaned)
    if balanced:
        candidates.append(balanced)

    last_error: json.JSONDecodeError | None = None
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as error:
            last_error = error
            nested = _extract_balanced_json(candidate)
            if nested and nested != candidate:
                try:
                    return json.loads(nested)
                except json.JSONDecodeError as nested_error:
                    last_error = nested_error
    raise LLMError("LLM response does not contain valid JSON") from last_error


__all__ = [
    "CompletionClient",
    "CompletionResult",
    "LLMError",
    "LLMHTTPError",
    "OpenAICompatibleClient",
    "parse_json_response",
]
