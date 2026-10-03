from __future__ import annotations

import unittest
from pathlib import Path

from eigen.config import load_project
from eigen.errors import CandidateError
from eigen.profiles import load_profile
from eigen.rendering import render_experiment
from eigen.search_space import SearchSpace
from eigen.tuning import ProfilePartitioner

PG_SERVER_PARAMETERS = {
    "postgres.shared_buffers_mb",
    "postgres.effective_cache_size_mb",
    "postgres.maintenance_work_mem_mb",
    "postgres.max_wal_size_mb",
    "postgres.work_mem_mb",
    "postgres.effective_io_concurrency",
    "postgres.random_page_cost",
    "postgres.max_parallel_workers",
    "postgres.max_parallel_maintenance_workers",
}


class PgvectorNativeProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = load_profile("pgvector-native-dense")
        self.space = SearchSpace(self.profile)

    def test_native_profile_has_three_conditional_regions(self) -> None:
        self.assertEqual(self.profile.capabilities.indexes, ["hnsw", "ivfflat"])
        self.assertEqual(self.profile.search_space.partition_by, ["index.type"])
        self.assertEqual(
            self.profile.search_space.parameters["index.type"].choices,
            ["exact", "hnsw", "ivfflat"],
        )

        regions = ProfilePartitioner(self.space).regions
        self.assertEqual(len(regions), 3)
        active = {region.fixed["index.type"]: set(region.active_parameters) for region in regions}
        self.assertEqual(
            active["exact"],
            {
                "index.type",
                "exact.max_parallel_workers_per_gather",
                *PG_SERVER_PARAMETERS,
            },
        )
        self.assertEqual(
            active["hnsw"],
            {
                "index.type",
                "hnsw.m",
                "hnsw.ef_construction",
                "hnsw.ef_search",
                *PG_SERVER_PARAMETERS,
            },
        )
        self.assertEqual(
            active["ivfflat"],
            {
                "index.type",
                "ivfflat.lists",
                "ivfflat.probes",
                *PG_SERVER_PARAMETERS,
            },
        )

    def test_each_native_region_renders_only_its_adapter_parameters(self) -> None:
        exact = render_experiment(self.profile, {"index.type": "exact"})
        self.assertEqual(exact["upload_params"]["index_type"], "exact")
        self.assertEqual(
            exact["search_params"][0]["config"],
            {"index_type": "exact", "exact_parallel_workers": 4},
        )
        self.assertEqual(exact["upload_params"]["hnsw_config"], {})
        self.assertEqual(exact["upload_params"]["ivfflat_config"], {})

        hnsw = render_experiment(
            self.profile,
            {
                "index.type": "hnsw",
                "hnsw.m": 24,
                "hnsw.ef_construction": 192,
                "hnsw.ef_search": 256,
            },
        )
        self.assertEqual(
            hnsw["upload_params"]["hnsw_config"],
            {"m": 24, "ef_construct": 192},
        )
        self.assertEqual(
            hnsw["search_params"][0]["config"],
            {"index_type": "hnsw", "hnsw_ef": 256},
        )
        self.assertEqual(hnsw["upload_params"]["ivfflat_config"], {})

        ivfflat = render_experiment(
            self.profile,
            {
                "index.type": "ivfflat",
                "ivfflat.lists": 4096,
                "ivfflat.probes": 128,
            },
        )
        self.assertEqual(ivfflat["upload_params"]["ivfflat_config"], {"lists": 4096})
        self.assertEqual(
            ivfflat["search_params"][0]["config"],
            {"index_type": "ivfflat", "ivfflat_probes": 128},
        )
        self.assertEqual(ivfflat["upload_params"]["hnsw_config"], {})

    def test_inactive_and_relational_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(CandidateError, "inactive"):
            self.space.canonicalize({"index.type": "exact", "hnsw.m": 16})
        with self.assertRaisesRegex(CandidateError, "outside"):
            self.space.canonicalize(
                {
                    "index.type": "ivfflat",
                    "ivfflat.lists": 32,
                    "ivfflat.probes": 64,
                }
            )
        with self.assertRaisesRegex(CandidateError, "hnsw-ef-search-at-least-top-k"):
            self.space.canonicalize(
                {"index.type": "hnsw", "hnsw.ef_search": 32},
                runtime={"top_k": 40},
            )
        with self.assertRaisesRegex(CandidateError, "parallel-maintenance-workers-within-total"):
            self.space.canonicalize(
                {
                    "postgres.max_parallel_workers": 8,
                    "postgres.max_parallel_maintenance_workers": 15,
                }
            )

    def test_system_defaults_render_into_server_configuration(self) -> None:
        experiment = render_experiment(self.profile, {"index.type": "hnsw"})
        settings = experiment["server_params"]["postgresql"]
        self.assertEqual(settings["shared_buffers_mb"], 12288)
        self.assertEqual(settings["effective_cache_size_mb"], 30720)
        self.assertEqual(settings["maintenance_work_mem_mb"], 20480)
        self.assertEqual(settings["max_wal_size_mb"], 8192)
        self.assertEqual(settings["max_parallel_workers"], 16)
        self.assertEqual(settings["max_parallel_maintenance_workers"], 15)

    def test_ann_profile_has_exactly_two_non_exact_regions(self) -> None:
        profile = load_profile("pgvector-ann-dense")
        space = SearchSpace(profile)
        regions = ProfilePartitioner(space).regions
        self.assertEqual(len(regions), 2)
        self.assertEqual(
            {region.fixed["index.type"] for region in regions},
            {"hnsw", "ivfflat"},
        )
        self.assertNotIn(
            "exact.max_parallel_workers_per_gather",
            profile.search_space.parameters,
        )
        with self.assertRaisesRegex(CandidateError, "not one of"):
            space.canonicalize({"index.type": "exact"})

    def test_paper_examples_use_ann_profile_and_fresh_builds(self) -> None:
        path = Path(__file__).resolve().parents[1] / "examples/paper/pgvector-mini-00.json"
        project = load_project(path)
        self.assertEqual(project.profile.id, "pgvector-ann-dense")
        self.assertEqual(project.config.tuning.strategy, "calm")
        self.assertFalse(project.config.runner.settings["state_reuse"])
        self.assertEqual([o.metric for o in project.config.tuning.objectives], ["qps"])
        self.assertTrue(Path(project.config.runner.settings["dataset_path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
