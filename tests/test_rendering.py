from __future__ import annotations

import unittest

from mutune.errors import CandidateError
from mutune.models import EngineProfile
from mutune.profiles import load_profile
from mutune.rendering import ExperimentRenderer, get_json_pointer, render_experiment

CANDIDATE = {
    "hnsw.m": 32,
    "hnsw.ef_construction": 256,
    "hnsw.ef_search": 512,
}


class GoldenRenderingTests(unittest.TestCase):
    def test_milvus_golden_experiment(self) -> None:
        experiment = render_experiment(
            load_profile("milvus-hnsw-dense"),
            CANDIDATE,
            {"experiment_name": "trial-milvus"},
        )
        self.assertEqual(
            experiment,
            {
                "name": "trial-milvus",
                "engine": "milvus",
                "connection_params": {},
                "collection_params": {},
                "upload_params": {
                    "parallel": 16,
                    "index_type": "HNSW",
                    "index_params": {"M": 32, "efConstruction": 256},
                },
                "search_params": [{"parallel": 1, "top": 10, "config": {"ef": 512}}],
            },
        )

    def test_qdrant_golden_experiment(self) -> None:
        experiment = render_experiment(
            load_profile("qdrant-hnsw-dense"),
            CANDIDATE,
            {
                "experiment_name": "trial-qdrant",
                "connection_params": {"timeout": 90, "https": True},
            },
        )
        self.assertEqual(
            experiment,
            {
                "name": "trial-qdrant",
                "engine": "qdrant",
                "connection_params": {"timeout": 90, "https": True},
                "collection_params": {"hnsw_config": {"m": 32, "ef_construct": 256}},
                "upload_params": {"parallel": 16, "batch_size": 1024},
                "search_params": [
                    {
                        "parallel": 1,
                        "top": 10,
                        "config": {"hnsw_ef": 512},
                    }
                ],
            },
        )

    def test_pgvector_golden_experiment(self) -> None:
        experiment = render_experiment(
            load_profile("pgvector-hnsw-dense"),
            CANDIDATE,
            {
                "experiment_name": "trial-pgvector",
                "upload_parallel": 4,
                "search_parallel": 8,
                "batch_size": 512,
                "top_k": 20,
            },
        )
        self.assertEqual(
            experiment,
            {
                "name": "trial-pgvector",
                "engine": "pgvector",
                "connection_params": {},
                "collection_params": {},
                "upload_params": {
                    "parallel": 4,
                    "batch_size": 512,
                    "hnsw_config": {"m": 32, "ef_construct": 256},
                },
                "search_params": [
                    {
                        "parallel": 8,
                        "top": 20,
                        "config": {"hnsw_ef": 512},
                    }
                ],
            },
        )


class RenderingBehaviorTests(unittest.TestCase):
    def test_rendering_is_deep_copy_isolated(self) -> None:
        profile = load_profile("qdrant-hnsw-dense")
        renderer = ExperimentRenderer(profile)
        first = renderer.render(CANDIDATE)
        first["collection_params"]["hnsw_config"]["m"] = 999
        first["connection_params"]["nested"] = {"changed": True}

        second = renderer.render(CANDIDATE)
        self.assertEqual(second["collection_params"]["hnsw_config"]["m"], 32)
        self.assertNotIn("nested", second["connection_params"])
        self.assertEqual(profile.experiment.template["collection_params"], {"hnsw_config": {}})

    def test_runtime_merge_does_not_remove_template_defaults(self) -> None:
        experiment = render_experiment(
            load_profile("qdrant-hnsw-dense"),
            CANDIDATE,
            {"connection_params": {"https": True}},
        )
        self.assertEqual(experiment["connection_params"], {"timeout": 30, "https": True})

    def test_unknown_runtime_and_runtime_constraint_are_rejected(self) -> None:
        profile = load_profile("milvus-hnsw-dense")
        with self.assertRaisesRegex(CandidateError, "unknown runtime"):
            render_experiment(profile, CANDIDATE, {"typo": 1})
        with self.assertRaisesRegex(CandidateError, "ef-search-at-least-top-k"):
            render_experiment(
                profile,
                {**CANDIDATE, "hnsw.ef_search": 16},
                {"top_k": 20},
            )

    def test_json_pointer_reads_nested_result(self) -> None:
        payload = {"results": {"rps": 123.5}}
        self.assertEqual(get_json_pointer(payload, "/results/rps"), 123.5)
        with self.assertRaises(KeyError):
            get_json_pointer(payload, "/results/missing")

    def test_value_map_can_render_a_native_json_fragment(self) -> None:
        payload = load_profile("qdrant-hnsw-dense").model_dump(mode="json")
        payload["search_space"]["parameters"]["quantization.type"] = {
            "kind": "categorical",
            "default": "none",
            "choices": ["none", "scalar"],
            "effect": "collection",
            "bindings": [
                {
                    "pointer": "/collection_params/quantization_config",
                    "value_map": {
                        "none": None,
                        "scalar": {
                            "scalar": {"type": "int8", "quantile": 0.99, "always_ram": False}
                        },
                    },
                }
            ],
        }
        profile = EngineProfile.model_validate(payload)
        rendered = render_experiment(
            profile,
            {**CANDIDATE, "quantization.type": "scalar"},
        )
        self.assertEqual(
            rendered["collection_params"]["quantization_config"],
            {
                "scalar": {
                    "type": "int8",
                    "quantile": 0.99,
                    "always_ram": False,
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
