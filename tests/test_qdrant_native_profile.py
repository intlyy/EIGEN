from __future__ import annotations

import unittest

from mutune.errors import CandidateError
from mutune.profiles import load_profile
from mutune.rendering import render_experiment
from mutune.search_space import SearchSpace
from mutune.tuning import ProfilePartitioner


class QdrantNativeProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_profile("qdrant-native-dense")
        self.space = SearchSpace(self.profile)

    def test_declares_only_the_real_dense_ann_family(self) -> None:
        self.assertEqual(self.profile.capabilities.indexes, ["hnsw"])
        self.assertEqual(
            self.profile.search_space.partition_by,
            ["index.mode", "search.mode", "quantization.type"],
        )

    def test_plain_mode_disables_indexing_and_prunes_hnsw_parameters(self) -> None:
        candidate = self.space.canonicalize({"index.mode": "plain"})
        self.assertEqual(
            candidate,
            {
                "index.mode": "plain",
                "vectors.on_disk": False,
                "optimizer.default_segments": 2,
                "payload.on_disk": False,
                "optimizer.max_segment_size": 200000,
                "optimizer.memmap_threshold": 200000,
                "optimizer.flush_interval_sec": 5,
                "optimizer.max_optimization_threads": 2,
            },
        )

        experiment = render_experiment(self.profile, candidate)
        self.assertEqual(
            experiment["collection_params"]["optimizers_config"]["indexing_threshold"],
            0,
        )
        self.assertEqual(experiment["search_params"][0]["config"], {"exact": True})
        self.assertNotIn("quantization_config", experiment["collection_params"])
        self.assertEqual(experiment["collection_params"]["hnsw_config"], {})

    def test_hnsw_scalar_quantization_renders_adapter_native_shapes(self) -> None:
        candidate = {
            "index.mode": "hnsw",
            "search.mode": "approximate",
            "quantization.type": "scalar",
            "hnsw.m": 32,
            "hnsw.ef_construction": 256,
            "hnsw.full_scan_threshold": 4096,
            "hnsw.on_disk": True,
            "hnsw.ef_search": 512,
            "vectors.on_disk": True,
            "optimizer.default_segments": 4,
            "quantization.rescore": True,
            "quantization.oversampling": 3.0,
        }
        experiment = render_experiment(self.profile, candidate)

        self.assertEqual(
            experiment["collection_params"],
            {
                "hnsw_config": {
                    "m": 32,
                    "ef_construct": 256,
                    "full_scan_threshold": 4096,
                    "on_disk": True,
                    "max_indexing_threads": 0,
                    "payload_m": 16,
                    "inline_storage": False,
                },
                "optimizers_config": {
                    "indexing_threshold": 1,
                    "default_segment_number": 4,
                    "max_segment_size": 200000,
                    "memmap_threshold": 200000,
                    "flush_interval_sec": 5,
                    "max_optimization_threads": 2,
                },
                "vectors_config": {"on_disk": True},
                "on_disk_payload": False,
                "quantization_config": {
                    "scalar": {
                        "type": "int8",
                        "quantile": 0.99,
                        "always_ram": True,
                    }
                },
            },
        )
        self.assertEqual(
            experiment["search_params"][0]["config"],
            {
                "exact": False,
                "hnsw_ef": 512,
                "quantization": {"rescore": True, "oversampling": 3.0, "ignore": False},
            },
        )

    def test_all_quantization_types_render_valid_union_discriminators(self) -> None:
        expected_keys = {
            "scalar": "scalar",
            "product": "product",
            "binary": "binary",
        }
        for quantization_type, expected_key in expected_keys.items():
            with self.subTest(quantization_type=quantization_type):
                experiment = render_experiment(
                    self.profile,
                    {"quantization.type": quantization_type},
                )
                quantization = experiment["collection_params"]["quantization_config"]
                self.assertEqual(set(quantization), {expected_key})

        unquantized = render_experiment(
            self.profile,
            {"quantization.type": "none"},
        )
        self.assertIsNone(unquantized["collection_params"]["quantization_config"])

    def test_exact_hnsw_mode_is_unquantized_full_precision_search(self) -> None:
        with self.assertRaisesRegex(CandidateError, "inactive"):
            self.space.canonicalize(
                {
                    "index.mode": "hnsw",
                    "search.mode": "exact",
                    "quantization.type": "binary",
                }
            )

        candidate = self.space.canonicalize(
            {
                "index.mode": "hnsw",
                "search.mode": "exact",
            }
        )
        self.assertNotIn("hnsw.ef_search", candidate)
        self.assertNotIn("quantization.type", candidate)
        self.assertNotIn("quantization.rescore", candidate)
        self.assertNotIn("quantization.oversampling", candidate)
        experiment = render_experiment(self.profile, candidate)
        self.assertEqual(experiment["search_params"][0]["config"], {"exact": True})
        self.assertNotIn("quantization_config", experiment["collection_params"])

    def test_partitioning_folds_inactive_nested_selectors(self) -> None:
        regions = ProfilePartitioner(self.space).regions
        self.assertEqual(len(regions), 6)
        self.assertIn({"index.mode": "plain"}, [region.fixed for region in regions])
        self.assertIn(
            {"index.mode": "hnsw", "search.mode": "exact"},
            [region.fixed for region in regions],
        )
        self.assertIn(
            {
                "index.mode": "hnsw",
                "search.mode": "approximate",
                "quantization.type": "binary",
            },
            [region.fixed for region in regions],
        )


if __name__ == "__main__":
    unittest.main()
