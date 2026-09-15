from __future__ import annotations

import unittest

from mutune.errors import CandidateError
from mutune.profiles import load_profile
from mutune.rendering import render_experiment
from mutune.search_space import SearchSpace
from mutune.tuning.partitioning import ProfilePartitioner

INDEX_TYPES = {
    "flat": "FLAT",
    "ivf_flat": "IVF_FLAT",
    "ivf_sq8": "IVF_SQ8",
    "ivf_pq": "IVF_PQ",
    "hnsw": "HNSW",
    "scann": "SCANN",
    "autoindex": "AUTOINDEX",
}

DEFAULT_CONFIGS = {
    "flat": ({}, {}),
    "ivf_flat": ({"nlist": 128}, {"nprobe": 8}),
    "ivf_sq8": ({"nlist": 128}, {"nprobe": 8}),
    "ivf_pq": (
        {"nlist": 128, "m": 4, "nbits": 8},
        {"nprobe": 8},
    ),
    "hnsw": ({"M": 16, "efConstruction": 128}, {"ef": 128}),
    "scann": ({"nlist": 128}, {"nprobe": 8, "reorder_k": 500}),
    "autoindex": ({}, {}),
}

MILVUS_SERVER_PARAMETERS = {
    "milvus.segment_max_size_mb",
    "milvus.segment_seal_proportion",
    "milvus.auto_handoff",
    "milvus.auto_balance",
    "milvus.graceful_time_ms",
    "milvus.insert_buffer_size_bytes",
    "milvus.min_segment_size_to_enable_index",
}


class MilvusNativeProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_profile("milvus-native-dense")
        self.space = SearchSpace(self.profile)

    def test_profile_loads_and_partitions_every_dense_index_type(self) -> None:
        self.assertEqual(set(self.profile.capabilities.indexes), set(INDEX_TYPES))
        self.assertEqual(self.profile.search_space.partition_by, ["index.type"])
        self.assertEqual(
            set(self.profile.search_space.parameters["index.type"].choices or []),
            set(INDEX_TYPES),
        )

        regions = ProfilePartitioner(self.space, runtime={"vector_size": 384}).regions
        self.assertEqual(len(regions), len(INDEX_TYPES))
        self.assertEqual(
            {region.fixed["index.type"] for region in regions},
            set(INDEX_TYPES),
        )
        active_by_index = {
            region.fixed["index.type"]: set(region.active_parameters) for region in regions
        }
        self.assertEqual(active_by_index["flat"], {"index.type", *MILVUS_SERVER_PARAMETERS})
        self.assertEqual(
            active_by_index["autoindex"],
            {"index.type", *MILVUS_SERVER_PARAMETERS},
        )
        self.assertEqual(
            active_by_index["ivf_pq"],
            {
                "index.type",
                "index.nlist",
                "index.nprobe",
                "ivf_pq.m",
                "ivf_pq.nbits",
                *MILVUS_SERVER_PARAMETERS,
            },
        )
        self.assertEqual(
            active_by_index["hnsw"],
            {
                "index.type",
                "hnsw.m",
                "hnsw.ef_construction",
                "hnsw.ef_search",
                *MILVUS_SERVER_PARAMETERS,
            },
        )

    def test_paper_server_defaults_render_for_every_index_region(self) -> None:
        experiment = render_experiment(
            self.profile,
            {"index.type": "flat"},
        )
        settings = experiment["server_params"]["milvus"]
        self.assertEqual(settings["data_coord_segment_max_size_mb"], 512)
        self.assertEqual(settings["data_coord_segment_seal_proportion"], 0.23)
        self.assertTrue(settings["query_coord_auto_handoff"])
        self.assertTrue(settings["query_coord_auto_balance"])
        self.assertEqual(settings["common_graceful_time_ms"], 5000)
        self.assertEqual(settings["data_node_segment_insert_buf_size_bytes"], 16777216)
        self.assertEqual(settings["root_coord_min_segment_size_to_enable_index"], 1024)

    def test_each_region_renders_the_native_milvus_index_name(self) -> None:
        for canonical, native in INDEX_TYPES.items():
            with self.subTest(index_type=canonical):
                experiment = render_experiment(
                    self.profile,
                    {"index.type": canonical},
                    {
                        "experiment_name": f"trial-{canonical}",
                        "vector_size": 384,
                    },
                )
                self.assertEqual(experiment["upload_params"]["index_type"], native)
                expected_index, expected_search = DEFAULT_CONFIGS[canonical]
                self.assertEqual(experiment["upload_params"]["index_params"], expected_index)
                self.assertEqual(experiment["search_params"][0]["config"], expected_search)

    def test_index_specific_parameters_are_pruned_and_rendered(self) -> None:
        ivf_pq = render_experiment(
            self.profile,
            {
                "index.type": "ivf_pq",
                "index.nlist": 2048,
                "index.nprobe": 64,
                "ivf_pq.m": 48,
                "ivf_pq.nbits": 8,
            },
            {"vector_size": 384},
        )
        self.assertEqual(
            ivf_pq["upload_params"]["index_params"],
            {"nlist": 2048, "m": 48, "nbits": 8},
        )
        self.assertEqual(ivf_pq["search_params"][0]["config"], {"nprobe": 64})

        scann = render_experiment(
            self.profile,
            {
                "index.type": "scann",
                "index.nlist": 1024,
                "index.nprobe": 32,
                "scann.reorder_k": 500,
            },
        )
        self.assertEqual(scann["upload_params"]["index_params"], {"nlist": 1024})
        self.assertEqual(
            scann["search_params"][0]["config"],
            {"nprobe": 32, "reorder_k": 500},
        )

        flat = render_experiment(self.profile, {"index.type": "flat"})
        self.assertEqual(flat["upload_params"]["index_params"], {})
        self.assertEqual(flat["search_params"][0]["config"], {})

    def test_invalid_cross_region_and_relational_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(CandidateError, "inactive"):
            self.space.canonicalize({"index.type": "flat", "hnsw.m": 16})
        with self.assertRaisesRegex(CandidateError, "nprobe-not-greater-than-nlist"):
            self.space.canonicalize(
                {
                    "index.type": "ivf_flat",
                    "index.nlist": 8,
                    "index.nprobe": 16,
                }
            )
        with self.assertRaisesRegex(CandidateError, "ivf-pq-m-divides-vector-size"):
            self.space.canonicalize(
                {"index.type": "ivf_pq", "ivf_pq.m": 16},
                runtime={"vector_size": 100},
            )


if __name__ == "__main__":
    unittest.main()
