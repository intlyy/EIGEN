from __future__ import annotations

import json
import random
import unittest

from pydantic import ValidationError

from mutune.api import WorkloadSpec
from mutune.errors import CandidateError, ConfigurationError
from mutune.models import EngineProfile, SearchSpaceSpec
from mutune.profiles import (
    list_builtin_profiles,
    load_profile,
    profile_fingerprint,
    validate_workload,
)
from mutune.search_space import SearchSpace

COMMON_PROFILE_IDS = {
    "milvus-hnsw-dense",
    "qdrant-hnsw-dense",
    "pgvector-hnsw-dense",
}
NATIVE_PROFILE_IDS = {
    "milvus-native-dense",
    "qdrant-native-dense",
    "pgvector-ann-dense",
    "pgvector-native-dense",
}
PROFILE_IDS = COMMON_PROFILE_IDS | NATIVE_PROFILE_IDS


class BuiltinProfileTests(unittest.TestCase):
    def test_all_builtin_profiles_load_strictly(self) -> None:
        self.assertEqual(set(list_builtin_profiles()), PROFILE_IDS)
        for profile_id in PROFILE_IDS:
            with self.subTest(profile_id=profile_id):
                profile = load_profile(profile_id)
                self.assertEqual(profile.id, profile_id)
                self.assertEqual(profile.adapter.plugin, "vector-db-benchmark")
                if profile_id in COMMON_PROFILE_IDS:
                    self.assertEqual(profile.search_space.domain, "common-hnsw-v1")
                    self.assertEqual(
                        set(profile.search_space.parameters),
                        {"hnsw.m", "hnsw.ef_construction", "hnsw.ef_search"},
                    )
                else:
                    self.assertTrue(profile.search_space.partition_by)

    def test_extra_fields_are_rejected(self) -> None:
        payload = load_profile("milvus-hnsw-dense").model_dump(mode="json")
        payload["misspelled"] = True
        with self.assertRaises(ValidationError):
            EngineProfile.model_validate_json(json.dumps(payload))

    def test_activation_cycles_are_rejected(self) -> None:
        payload = load_profile("milvus-hnsw-dense").model_dump(mode="json")
        parameters = payload["search_space"]["parameters"]
        parameters["hnsw.m"]["active_if"] = [
            {"parameter": "hnsw.ef_search", "op": "ne", "value": 0}
        ]
        parameters["hnsw.ef_search"]["active_if"] = [
            {"parameter": "hnsw.m", "op": "ne", "value": 0}
        ]
        with self.assertRaisesRegex(ValidationError, "activation dependency cycle"):
            EngineProfile.model_validate_json(json.dumps(payload))

    def test_parent_child_binding_collisions_are_rejected(self) -> None:
        payload = load_profile("milvus-hnsw-dense").model_dump(mode="json")
        payload["search_space"]["parameters"]["hnsw.m"]["bindings"] = [
            {"pointer": "/upload_params/index_params"}
        ]
        with self.assertRaisesRegex(ValidationError, "binding collision"):
            EngineProfile.model_validate_json(json.dumps(payload))

    def test_server_bindings_and_effect_must_agree(self) -> None:
        payload = load_profile("milvus-hnsw-dense").model_dump(mode="json")
        parameter = payload["search_space"]["parameters"]["hnsw.m"]
        parameter["effect"] = "server"
        with self.assertRaisesRegex(ValidationError, "must bind below /server_params"):
            EngineProfile.model_validate(payload)

        payload = load_profile("milvus-hnsw-dense").model_dump(mode="json")
        parameter = payload["search_space"]["parameters"]["hnsw.m"]
        parameter["bindings"] = [{"pointer": "/server_params/milvus/m"}]
        with self.assertRaisesRegex(ValidationError, "must have effect='server'"):
            EngineProfile.model_validate(payload)

    def test_categorical_value_map_must_cover_every_choice(self) -> None:
        payload = load_profile("milvus-hnsw-dense").model_dump(mode="json")
        payload["search_space"]["parameters"]["index.type"] = {
            "kind": "categorical",
            "default": "hnsw",
            "choices": ["hnsw", "flat"],
            "effect": "index",
            "bindings": [
                {
                    "pointer": "/upload_params/index_type",
                    "value_map": {"hnsw": "HNSW"},
                }
            ],
        }
        with self.assertRaisesRegex(ValidationError, "exactly match"):
            EngineProfile.model_validate(payload)

    def test_profile_fingerprint_is_stable_and_profile_specific(self) -> None:
        milvus = load_profile("milvus-hnsw-dense")
        qdrant = load_profile("qdrant-hnsw-dense")
        self.assertEqual(profile_fingerprint(milvus), profile_fingerprint(milvus))
        self.assertNotEqual(profile_fingerprint(milvus), profile_fingerprint(qdrant))

    def test_unknown_builtin_profile_is_clear(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "unknown built-in profile"):
            load_profile("does-not-exist")


class WorkloadCapabilityTests(unittest.TestCase):
    def make_workload(self, **overrides: object) -> WorkloadSpec:
        values = {
            "dataset": "tiny-1m-384-euclidean",
            "distance": "euclidean",
            "top_k": 10,
            "concurrency": 1,
            "vector_size": 384,
            "filtered": False,
            "sparse": False,
        }
        values.update(overrides)
        return WorkloadSpec(**values)  # type: ignore[arg-type]

    def test_supported_alias_and_filtered_workloads(self) -> None:
        validate_workload(
            load_profile("milvus-hnsw-dense"),
            self.make_workload(distance="angular", filtered=True),
        )
        validate_workload(
            load_profile("qdrant-hnsw-dense"),
            self.make_workload(distance="dot", filtered=True),
        )

    def test_pgvector_rejects_unsupported_dot_and_filtering(self) -> None:
        profile = load_profile("pgvector-hnsw-dense")
        with self.assertRaisesRegex(ConfigurationError, "distance"):
            validate_workload(profile, self.make_workload(distance="dot"))
        with self.assertRaisesRegex(ConfigurationError, "filtered"):
            validate_workload(profile, self.make_workload(filtered=True))

    def test_common_dense_profiles_reject_sparse_workloads(self) -> None:
        for profile_id in PROFILE_IDS:
            with self.subTest(profile_id=profile_id):
                with self.assertRaisesRegex(ConfigurationError, "sparse"):
                    validate_workload(load_profile(profile_id), self.make_workload(sparse=True))


class SearchSpaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = SearchSpace(load_profile("milvus-hnsw-dense"))

    def test_defaults_and_partial_candidate_are_canonicalized(self) -> None:
        self.assertEqual(
            self.space.canonicalize({"hnsw.m": 32}),
            {
                "hnsw.m": 32,
                "hnsw.ef_construction": 128,
                "hnsw.ef_search": 128,
            },
        )

    def test_unknown_out_of_range_and_constraint_failures(self) -> None:
        with self.assertRaisesRegex(CandidateError, "unknown candidate"):
            self.space.canonicalize({"hnsw.unknown": 1})
        with self.assertRaisesRegex(CandidateError, "outside"):
            self.space.canonicalize({"hnsw.m": 65})
        with self.assertRaisesRegex(CandidateError, "ef-search-at-least-top-k"):
            self.space.canonicalize({"hnsw.ef_search": 16}, runtime={"top_k": 20})

    def test_sampling_is_deterministic_and_valid(self) -> None:
        left_rng = random.Random(42)
        right_rng = random.Random(42)
        left = [self.space.sample(left_rng) for _ in range(25)]
        right = [self.space.sample(right_rng) for _ in range(25)]
        self.assertEqual(left, right)
        for candidate in left:
            self.assertEqual(candidate, self.space.validate(candidate))

    def test_effects_distinguish_search_from_index_rebuild(self) -> None:
        default = self.space.canonicalize({})
        search_change = {**default, "hnsw.ef_search": 256}
        index_change = {**default, "hnsw.m": 32}
        self.assertEqual(self.space.effects_between(default, search_change), frozenset({"search"}))
        self.assertEqual(self.space.effects_between(default, index_change), frozenset({"index"}))

    def test_conditional_activation_prunes_and_rejects_explicit_inactive_values(self) -> None:
        spec = SearchSpaceSpec.model_validate_json(
            json.dumps(
                {
                    "domain": "conditional-test",
                    "partition_by": ["index.family"],
                    "parameters": {
                        "index.family": {
                            "kind": "categorical",
                            "default": "hnsw",
                            "choices": ["hnsw", "flat"],
                            "effect": "index",
                            "bindings": [{"pointer": "/upload_params/index_type"}],
                        },
                        "hnsw.m": {
                            "kind": "integer",
                            "default": 16,
                            "bounds": [4, 64],
                            "step": 1,
                            "effect": "index",
                            "active_if": [
                                {
                                    "parameter": "index.family",
                                    "op": "eq",
                                    "value": "hnsw",
                                }
                            ],
                            "bindings": [{"pointer": "/upload_params/index_params/M"}],
                        },
                    },
                }
            )
        )
        space = SearchSpace(spec)
        self.assertEqual(
            space.canonicalize({"index.family": "flat"}),
            {"index.family": "flat"},
        )
        with self.assertRaisesRegex(CandidateError, "inactive"):
            space.canonicalize({"index.family": "flat", "hnsw.m": 16})

    def test_partition_keys_are_explicit_and_categorical(self) -> None:
        payload = {
            "domain": "bad-partition",
            "partition_by": ["hnsw.m"],
            "parameters": {
                "hnsw.m": {
                    "kind": "integer",
                    "default": 16,
                    "bounds": [4, 64],
                    "effect": "index",
                    "bindings": [{"pointer": "/upload_params/index_params/M"}],
                }
            },
        }
        with self.assertRaisesRegex(ValidationError, "categorical or boolean"):
            SearchSpaceSpec.model_validate(payload)

    def test_constraint_only_runtime_supports_divisibility(self) -> None:
        payload = load_profile("milvus-hnsw-dense").model_dump(mode="json")
        payload["experiment"]["runtime_context"] = ["vector_size"]
        payload["search_space"]["constraints"].append(
            {
                "id": "m-divides-vector-size",
                "left": {"source": "candidate", "key": "hnsw.m"},
                "op": "divides",
                "right": {"source": "runtime", "key": "vector_size"},
                "message": "m must divide vector_size",
            }
        )
        space = SearchSpace(EngineProfile.model_validate(payload))
        self.assertEqual(
            space.canonicalize({"hnsw.m": 32}, runtime={"vector_size": 384})["hnsw.m"],
            32,
        )
        with self.assertRaisesRegex(CandidateError, "m-divides-vector-size"):
            space.canonicalize({"hnsw.m": 5}, runtime={"vector_size": 384})


if __name__ == "__main__":
    unittest.main()
