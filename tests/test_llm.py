from __future__ import annotations

import os
import random
import unittest
from unittest.mock import patch
from urllib.error import URLError

from eigen.config import LLMConfig
from eigen.llm import (
    CompletionResult,
    LLMError,
    OpenAICompatibleClient,
    parse_json_response,
)
from eigen.profiles import load_profile
from eigen.search_space import SearchSpace
from eigen.tuning.history import candidate_key
from eigen.tuning.partitioning import Region
from eigen.tuning.proposer import HybridProposer, OpenAICompatibleProposer, RandomProposer


class JsonResponseTests(unittest.TestCase):
    def test_parses_think_fence_and_nested_json(self) -> None:
        response = (
            '<think>private reasoning with {"wrong": true}</think>\n'
            'Result:\n```json\n[{"hnsw.m": 16, "nested": {"x": [1, 2]}}]\n```'
        )
        self.assertEqual(
            parse_json_response(response),
            [{"hnsw.m": 16, "nested": {"x": [1, 2]}}],
        )

    def test_extracts_balanced_json_from_prose(self) -> None:
        self.assertEqual(
            parse_json_response('prefix {"text": "a } bracket", "items": [1, 2]} suffix'),
            {"text": "a } bracket", "items": [1, 2]},
        )

    def test_rejects_python_literal_instead_of_rewriting_quotes(self) -> None:
        with self.assertRaises(LLMError):
            parse_json_response("[{'hnsw.m': 16}]")


class OpenAICompatibleClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LLMConfig(
            model="test-model",
            base_url="https://llm.invalid/v1",
            api_key_env="EIGEN_TEST_API_KEY",
            max_tokens=100,
            timeout_s=5,
        )

    def test_credential_is_read_only_from_environment(self) -> None:
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update(
                url=url,
                authorization=headers["Authorization"],
                payload=payload,
                timeout=timeout,
            )
            return {
                "choices": [{"message": {"content": "[]"}}],
                "usage": {"total_tokens": 12},
            }

        client = OpenAICompatibleClient(self.config, transport=transport)
        with patch.dict(os.environ, {"EIGEN_TEST_API_KEY": "environment-secret"}, clear=False):
            result = client.complete("give JSON")
        self.assertEqual(result, CompletionResult(content="[]", usage={"total_tokens": 12}))
        self.assertEqual(captured["authorization"], "Bearer environment-secret")
        self.assertEqual(captured["url"], "https://llm.invalid/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "test-model")

    def test_missing_environment_credential_fails_before_transport(self) -> None:
        calls = []
        client = OpenAICompatibleClient(
            self.config,
            transport=lambda *args: calls.append(args),
        )
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(LLMError):
            client.complete("give JSON")
        self.assertEqual(calls, [])

    def test_nullable_sampling_fields_are_omitted_for_thinking_models(self) -> None:
        captured = {}
        config = LLMConfig(
            model="thinking-model",
            base_url="https://llm.invalid/v1",
            api_key_env="EIGEN_TEST_API_KEY",
            temperature=None,
            max_tokens=None,
            extra_body={"thinking": {"type": "enabled"}},
        )

        def transport(_url, _headers, payload, _timeout):
            captured.update(payload)
            return {"choices": [{"message": {"content": "[]"}}]}

        client = OpenAICompatibleClient(config, transport=transport)
        with patch.dict(
            os.environ,
            {"EIGEN_TEST_API_KEY": "environment-secret"},
            clear=False,
        ):
            client.complete("give JSON")

        self.assertNotIn("temperature", captured)
        self.assertNotIn("max_tokens", captured)
        self.assertEqual(captured["thinking"], {"type": "enabled"})

    def test_transient_failure_has_a_finite_attempt_limit(self) -> None:
        calls = []

        def failing_transport(*args):
            calls.append(args)
            raise URLError("temporary")

        client = OpenAICompatibleClient(
            self.config,
            max_attempts=3,
            retry_base_s=0,
            transport=failing_transport,
            sleep=lambda _: None,
        )
        with patch.dict(os.environ, {"EIGEN_TEST_API_KEY": "secret"}, clear=False):
            with self.assertRaisesRegex(LLMError, "3 attempts"):
                client.complete("give JSON")
        self.assertEqual(len(calls), 3)


class _StaticCompletionClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def complete(self, prompt: str, *, deadline=None) -> CompletionResult:
        self.calls += 1
        return CompletionResult(self.content)


class LLMProposerTests(unittest.TestCase):
    def test_hybrid_fills_valid_unseen_candidates_after_llm_deadline_expires(self) -> None:
        clock = [0.0]

        class TimeoutClient:
            def complete(self, prompt, *, deadline=None):
                clock[0] = deadline
                raise LLMError("LLM request deadline expired")

        space = SearchSpace(load_profile("qdrant-hnsw-dense"))
        primary = OpenAICompatibleProposer(
            space,
            TimeoutClient(),
            objective_metric="qps",
            constraint_metric="recall",
            constraint_threshold=0.9,
            monotonic=lambda: clock[0],
        )
        proposer = HybridProposer(primary, RandomProposer(space, rng=random.Random(42)))
        excluded = {candidate_key(space.canonicalize({}))}
        with patch("eigen.tuning.proposer.time.monotonic", side_effect=lambda: clock[0]):
            candidates = proposer.propose(
                4, region=Region("all"), history=(), excluded_keys=excluded, deadline=1.0
            )
        self.assertEqual(len(candidates), 4)
        self.assertEqual(len({candidate_key(c) for c in candidates}), 4)
        self.assertTrue(all(candidate_key(c) not in excluded for c in candidates))
        self.assertTrue(all(space.canonicalize(c) == c for c in candidates))

    def test_proposer_canonicalizes_and_deduplicates(self) -> None:
        profile = load_profile("milvus-hnsw-dense")
        space = SearchSpace(profile)
        client = _StaticCompletionClient(
            """```json
            [
              {"hnsw.m": 20, "hnsw.ef_construction": 200, "hnsw.ef_search": 300},
              {"hnsw.m": 20, "hnsw.ef_construction": 200, "hnsw.ef_search": 300},
              {"hnsw.m": 24, "hnsw.ef_construction": 220, "hnsw.ef_search": 320}
            ]
            ```"""
        )
        proposer = OpenAICompatibleProposer(
            space,
            client,
            runtime={"top_k": 10},
            objective_metric="qps",
            constraint_metric="recall",
            constraint_threshold=0.9,
            max_attempts=1,
        )
        candidates = proposer.propose(
            3,
            region=Region("all"),
            history=(),
            excluded_keys=set(),
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["hnsw.m"], 20)
        self.assertEqual(candidates[1]["hnsw.m"], 24)

    def test_invalid_json_stops_after_configured_attempts(self) -> None:
        profile = load_profile("milvus-hnsw-dense")
        client = _StaticCompletionClient("not JSON")
        proposer = OpenAICompatibleProposer(
            SearchSpace(profile),
            client,
            objective_metric="qps",
            constraint_metric="recall",
            constraint_threshold=0.9,
            max_attempts=2,
        )
        self.assertEqual(
            proposer.propose(
                1,
                region=Region("all"),
                history=(),
                excluded_keys=set(),
            ),
            [],
        )
        self.assertEqual(client.calls, 2)


if __name__ == "__main__":
    unittest.main()
