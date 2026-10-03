from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from eigen.geo_minidb import build_geo_minidbs
from eigen.minidb import (
    ALLOCATION_VERSION,
    build_hdf5_minidbs,
    exact_ground_truth,
    hash_bucketize,
    proportional_allocation,
    sampling_partition,
)


class MiniDBTests(unittest.TestCase):
    def test_geo_builder_records_cost_and_preserves_short_and_empty_exact_answers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            np.save(source / "vectors.npy", np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
            (source / "payloads.jsonl").write_text(
                "".join(
                    json.dumps({"location": {"lat": lat, "lon": 0}}) + "\n" for lat in (0, 10, 20)
                )
            )
            rows = [
                {
                    "query": [1, 0],
                    "conditions": {
                        "and": [{"location": {"geo": {"lat": lat, "lon": 0, "radius": 10}}}]
                    },
                }
                for lat in (0, 80)
            ]
            (source / "tests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest = build_geo_minidbs(
                source, root / "views", sample_ratio=1.0, sample_seeds=(1, 2), top_k=3
            )
            self.assertEqual(manifest["partition"]["family"], "angular-hyperplane")
            self.assertEqual(manifest["partition"]["metric"], "cosine")
            self.assertEqual(manifest["sampling_method"], "locality-stratified")
            self.assertEqual(manifest["allocation_version"], ALLOCATION_VERSION)
            timing = manifest["construction_timing"]
            self.assertGreater(timing["wall_s"], 0)
            self.assertAlmostEqual(timing["wall_s"], sum(timing["stages_wall_s"].values()))
            self.assertGreater(timing["stages_wall_s"]["ground_truth_and_query_write"], 0)
            for entry in manifest["minidbs"]:
                rows = [
                    json.loads(line)
                    for line in (root / "views" / entry["path"] / "tests.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual([len(row["closest_ids"]) for row in rows], [1, 0])

    def test_minimum_one_and_exact_size_with_tiny_buckets(self):
        np.testing.assert_array_equal(proportional_allocation(np.array([99, 1, 1]), 10), [8, 1, 1])
        np.testing.assert_array_equal(proportional_allocation(np.array([5, 5]), 3), [2, 1])
        with self.assertRaises(ValueError):
            proportional_allocation(np.array([5, 5]), 1)

    def test_reserve_one_before_apportioning_original_populations(self):
        # The old full-budget quotas yielded [10, 2]. The paper reserves two
        # first, then apportions the remaining ten exactly as [8, 2].
        np.testing.assert_array_equal(proportional_allocation(np.array([80, 20]), 12), [9, 3])
        # Equal fractional remainders are settled in stable bucket order.
        np.testing.assert_array_equal(proportional_allocation(np.array([5, 5, 5]), 5), [2, 2, 1])

    def test_capacity_saturation_redistributes_to_remaining_buckets(self):
        np.testing.assert_array_equal(proportional_allocation(np.array([2, 4, 94]), 97), [2, 4, 91])
        counts = np.array([1] * 100 + [1000])
        np.testing.assert_array_equal(proportional_allocation(counts, 150), [1] * 100 + [50])
        np.testing.assert_array_equal(proportional_allocation(counts, len(counts)), np.ones(101))
        np.testing.assert_array_equal(proportional_allocation(counts, int(counts.sum())), counts)

    def test_allocation_boundaries_conservation_reproducibility_and_input_invariance(self):
        rng = np.random.default_rng(23)
        for buckets in (1, 2, 7, 30):
            counts = rng.integers(1, 12, size=buckets)
            original = counts.copy()
            for size in range(buckets, int(counts.sum()) + 1):
                allocation = proportional_allocation(counts, size)
                self.assertEqual(int(allocation.sum()), size)
                self.assertTrue(np.all((allocation >= 1) & (allocation <= counts)))
                np.testing.assert_array_equal(allocation, proportional_allocation(counts, size))
                np.testing.assert_array_equal(counts, original)
        for invalid in (np.array([]), np.array([0, 2]), np.array([-1]), np.array([[2]])):
            with self.subTest(counts=invalid.tolist()), self.assertRaises(ValueError):
                proportional_allocation(invalid, 1)
        for invalid_size in (0, 1, 5, 2.5, True):
            with self.subTest(size=invalid_size), self.assertRaises(ValueError):
                proportional_allocation(np.array([2, 2]), invalid_size)

    def test_uniform_sampling_skips_lsh_and_is_seeded_without_replacement(self):
        vectors = np.random.default_rng(19).normal(size=(100, 4))
        original = vectors.copy()
        with patch("eigen.minidb.hash_bucketize", side_effect=AssertionError("LSH not needed")):
            buckets = sampling_partition(vectors, 12, sampling_method="uniform", metric="l2")
            other_seed = sampling_partition(
                vectors, 12, sampling_method="uniform", metric="l2", bucket_seed=999
            )
        self.assertEqual(buckets.family, "uniform-single-bucket")
        self.assertEqual(buckets.fingerprint, other_seed.fingerprint)
        self.assertIsNone(buckets.partition_metadata["projections_file"])
        self.assertIsNone(buckets.partition_metadata["offsets_file"])
        self.assertEqual(buckets.effective_bits, 0)
        for seed in (4, 5):
            expected = np.sort(np.random.default_rng(seed).choice(100, 12, replace=False))
            np.testing.assert_array_equal(buckets.sample(12, seed), expected)
            np.testing.assert_array_equal(other_seed.sample(12, seed), expected)
        self.assertFalse(np.array_equal(buckets.sample(12, 4), buckets.sample(12, 5)))
        np.testing.assert_array_equal(buckets.sample(100, 4), np.arange(100))
        np.testing.assert_array_equal(vectors, original)
        with self.assertRaises(ValueError):
            sampling_partition(vectors, 12, sampling_method="unknown")

    def test_shared_partition_adapts_bucket_count(self):
        vectors = np.random.default_rng(1).normal(size=(100, 12)).astype(np.float32)
        buckets = hash_bucketize(vectors, 5, num_hash_bits=12, bucket_seed=42)
        self.assertLessEqual(len(buckets.counts), 5)
        for seed in (10, 11, 12):
            ids = buckets.sample(5, seed)
            self.assertEqual(len(set(ids)), 5)
            self.assertEqual(set(buckets.codes[ids]), set(buckets.codes))
        np.testing.assert_array_equal(buckets.sample(5, 10), buckets.sample(5, 10))

    def test_l2_partition_distinguishes_lengths_while_cosine_preserves_directions(self):
        vectors = np.array([[1, 0], [5, 0], [10, 0]], dtype=np.float32)
        euclidean = hash_bucketize(vectors, 3, metric="euclidean", l2_bucket_width=1.0)
        angular = hash_bucketize(vectors, 3, metric="angular")
        self.assertEqual(len(euclidean.counts), 3)
        self.assertEqual(len(angular.counts), 1)
        self.assertEqual(euclidean.family, "euclidean-p-stable")
        self.assertEqual(angular.family, "angular-hyperplane")
        np.testing.assert_array_equal(vectors, [[1, 0], [5, 0], [10, 0]])

    def test_partition_uses_longest_admissible_shared_prefix(self):
        vectors = np.random.default_rng(1).normal(size=(100, 4))
        for metric in ("l2", "cosine"):
            with self.subTest(metric=metric):
                full = hash_bucketize(vectors, 100, metric=metric, l2_bucket_width=2.0)
                if metric == "l2":
                    hashes = np.floor((vectors @ full.planes + full.offsets) / 2.0)
                else:
                    hashes = vectors @ full.planes >= 0
                for size in (1, 5, 10, 30):
                    buckets = hash_bucketize(vectors, size, metric=metric, l2_bucket_width=2.0)
                    feasible = [
                        k for k in range(13) if len(np.unique(hashes[:, :k], axis=0)) <= size
                    ]
                    self.assertEqual(buckets.effective_bits, max(feasible))
                    np.testing.assert_array_equal(
                        buckets.planes, full.planes[:, : buckets.effective_bits]
                    )
                    # Bucket IDs may be compressed, but must identify exactly the
                    # prefix tuples, without collisions or extra subdivisions.
                    prefix = hashes[:, : buckets.effective_bits]
                    expected_equal = (prefix[:, None, :] == prefix[None, :, :]).all(axis=2)
                    np.testing.assert_array_equal(
                        buckets.codes[:, None] == buckets.codes[None, :], expected_equal
                    )
                    sampled = buckets.sample(size, 10)
                    self.assertEqual(len(np.unique(sampled)), size)
                    self.assertEqual(set(buckets.codes[sampled]), set(buckets.codes))
                    allocation = proportional_allocation(buckets.counts, size)
                    self.assertTrue(np.all((allocation >= 1) & (allocation <= buckets.counts)))
                    self.assertEqual(int(allocation.sum()), size)

    def test_l2_hash_sequence_and_views_are_reproducible(self):
        vectors = np.random.default_rng(1).normal(size=(100, 4))
        first = hash_bucketize(vectors, 30, metric="l2", l2_bucket_width=2.0, chunk_size=7)
        repeated = hash_bucketize(vectors, 30, metric="l2", l2_bucket_width=2.0)
        self.assertEqual(first.fingerprint, repeated.fingerprint)
        np.testing.assert_array_equal(first.sample(30, 10), repeated.sample(30, 10))
        self.assertFalse(np.array_equal(first.sample(30, 10), first.sample(30, 11)))
        shorter = hash_bucketize(vectors, 100, metric="l2", l2_bucket_width=2.0, num_hash_bits=2)
        np.testing.assert_array_equal(first.planes, shorter.planes)
        np.testing.assert_array_equal(first.offsets, shorter.offsets)
        different = hash_bucketize(vectors, 30, metric="l2", bucket_seed=43, l2_bucket_width=2.0)
        self.assertNotEqual(first.fingerprint, different.fingerprint)
        for width in (0, -1, float("inf"), float("nan")):
            with self.subTest(width=width), self.assertRaises(ValueError):
                hash_bucketize(vectors, 30, metric="l2", l2_bucket_width=width)

    def test_l2_empty_prefix_fallback_and_single_bucket_capacity(self):
        vectors = np.array([[1, 0], [5, 0], [10, 0]], dtype=np.float32)
        fallback = hash_bucketize(vectors, 1, metric="l2")
        self.assertEqual(fallback.effective_bits, 0)
        self.assertEqual(fallback.planes.shape, (2, 0))
        self.assertEqual(len(fallback.offsets), 0)
        np.testing.assert_array_equal(fallback.counts, [3])
        self.assertEqual(len(fallback.sample(1, 1)), 1)
        duplicates = np.tile([[1.0, 2.0]], (10, 1))
        bucket = hash_bucketize(duplicates, 3, metric="l2")
        self.assertEqual(bucket.effective_bits, 12)
        self.assertEqual(len(bucket.sample(3, 1)), 3)

    def test_exact_l2_ground_truth_uses_distances_not_squared_distances(self):
        train = np.array([[0, 0], [3, 0], [0, 4]], dtype=np.float32)
        distances, ids = exact_ground_truth(train, train[:1], "l2", 3)
        np.testing.assert_allclose(distances, [[0, 3, 4]])
        np.testing.assert_array_equal(ids, [[0, 1, 2]])

    def test_views_preserve_vectors_queries_and_recompute_local_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = np.random.default_rng(2).normal(size=(50, 4)).astype(np.float32)
            queries = original[:2].copy()
            source = root / "source.hdf5"
            with h5py.File(source, "w") as f:
                f["train"], f["test"], f["neighbors"] = original, queries, [[999], [999]]
            manifest = build_hdf5_minidbs(
                source, root / "views", sample_ratio=0.2, top_k=3, l2_bucket_width=2.5
            )
            self.assertEqual(len(manifest["minidbs"]), 3)
            self.assertEqual(sum(manifest["allocation"]), 10)
            self.assertEqual(manifest["method"], "shared-p-stable-stratified")
            self.assertEqual(manifest["sampling_method"], "locality-stratified")
            self.assertEqual(manifest["allocation_version"], ALLOCATION_VERSION)
            partition = manifest["partition"]
            self.assertEqual(partition["family"], "euclidean-p-stable")
            self.assertEqual(partition["l2_bucket_width"], 2.5)
            self.assertEqual(partition["effective_hash_functions"], manifest["effective_bits"])
            reproduced = hash_bucketize(original, 10, metric="l2", l2_bucket_width=2.5)
            self.assertEqual(reproduced.fingerprint, manifest["bucket_fingerprint"])
            np.testing.assert_array_equal(
                np.load(root / "views" / partition["projections_file"]), reproduced.planes
            )
            np.testing.assert_array_equal(
                np.load(root / "views" / partition["offsets_file"]), reproduced.offsets
            )
            timing = manifest["construction_timing"]
            self.assertGreater(timing["wall_s"], 0)
            self.assertAlmostEqual(timing["wall_s"], sum(timing["stages_wall_s"].values()))
            for stage in ("bucketization", "sampling", "ground_truth", "data_write", "checksums"):
                self.assertGreater(timing["stages_wall_s"][stage], 0)
            for entry in manifest["minidbs"]:
                path = root / "views" / entry["path"]
                ids = np.load(path.with_suffix(".indices.npy"))
                with h5py.File(path, "r") as f:
                    np.testing.assert_array_equal(f["train"][:], original[ids])
                    np.testing.assert_array_equal(f["test"][:], queries)
                    self.assertLess(f["neighbors"][:].max(), 10)
                    self.assertEqual(f.attrs["bucket_fingerprint"], manifest["bucket_fingerprint"])
                    expected = np.linalg.norm(
                        queries[:, None, :] - original[ids][None, :, :], axis=2
                    )
                    neighbors = np.argsort(expected, axis=1)[:, :3]
                    np.testing.assert_array_equal(f["neighbors"][:], neighbors)
                    np.testing.assert_allclose(
                        f["distances"][:],
                        np.take_along_axis(expected, neighbors, axis=1),
                        atol=1e-6,
                    )
            with self.assertRaises(FileExistsError):
                build_hdf5_minidbs(source, root / "views", top_k=3)

    def test_uniform_hdf5_views_preserve_workload_and_recompute_exact_neighbors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = np.random.default_rng(32).normal(size=(40, 4)).astype(np.float32)
            queries = original[:3].copy()
            source = root / "source.hdf5"
            with h5py.File(source, "w") as f:
                f["train"], f["test"], f["neighbors"] = original, queries, [[999]] * 3
            manifest = build_hdf5_minidbs(
                source,
                root / "views",
                sample_ratio=0.25,
                sampling_method="uniform",
                sample_seeds=(17, 18),
                top_k=3,
            )
            self.assertEqual(manifest["sampling_method"], "uniform")
            self.assertEqual(manifest["method"], "uniform-without-replacement")
            self.assertEqual(manifest["allocation_version"], ALLOCATION_VERSION)
            self.assertEqual(manifest["allocation"], [10])
            self.assertEqual(manifest["nonempty_buckets"], 1)
            self.assertEqual(manifest["requested_bits"], 0)
            self.assertTrue(manifest["bucket_fingerprint"])
            self.assertFalse((root / "views" / "hash_planes.npy").exists())
            self.assertFalse((root / "views" / "hash_offsets.npy").exists())
            for entry in manifest["minidbs"]:
                path = root / "views" / entry["path"]
                ids = np.load(path.with_suffix(".indices.npy"))
                expected_ids = np.sort(
                    np.random.default_rng(entry["sample_seed"]).choice(40, 10, replace=False)
                )
                np.testing.assert_array_equal(ids, expected_ids)
                with h5py.File(path, "r") as f:
                    self.assertEqual(f.attrs["sampling_method"], "uniform")
                    self.assertEqual(f.attrs["allocation_version"], ALLOCATION_VERSION)
                    np.testing.assert_array_equal(f["train"][:], original[ids])
                    np.testing.assert_array_equal(f["test"][:], queries)
                    distances = np.linalg.norm(queries[:, None] - original[ids][None, :], axis=2)
                    expected = np.argsort(distances, axis=1)[:, :3]
                    np.testing.assert_array_equal(f["neighbors"][:], expected)
                    np.testing.assert_allclose(
                        f["distances"][:],
                        np.take_along_axis(distances, expected, axis=1),
                        atol=1e-6,
                    )

    def test_uniform_geo_views_keep_filters_payloads_and_exact_local_answers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            vectors = np.array([[i, 1] for i in range(1, 9)], dtype=np.float32)
            np.save(source / "vectors.npy", vectors)
            payloads = [{"location": {"lat": 10 * (i % 2), "lon": 0}} for i in range(8)]
            (source / "payloads.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in payloads)
            )
            queries = [{"query": [1, 0]}] + [
                {
                    "query": [1, 0],
                    "conditions": {
                        "and": [{"location": {"geo": {"lat": lat, "lon": 0, "radius": 10}}}]
                    },
                }
                for lat in (0, 80)
            ]
            (source / "tests.jsonl").write_text(
                "".join(json.dumps({**row, "closest_ids": [999]}) + "\n" for row in queries)
            )
            manifest = build_geo_minidbs(
                source,
                root / "views",
                sample_ratio=0.75,
                sampling_method="uniform",
                sample_seeds=(7, 8),
                top_k=3,
            )
            self.assertEqual(manifest["sampling_method"], "uniform")
            self.assertEqual(manifest["method"], "uniform-without-replacement")
            self.assertEqual(manifest["allocation_version"], ALLOCATION_VERSION)
            self.assertEqual(manifest["allocation"], [6])
            self.assertEqual(manifest["partition"]["family"], "uniform-single-bucket")
            self.assertEqual(manifest["payload_schema"], {"location": "geo"})
            self.assertIsNone(manifest["partition"]["projections_file"])
            for entry in manifest["minidbs"]:
                path = root / "views" / entry["path"]
                ids = np.load(path / "sampled_indices.npy")
                expected_ids = np.sort(
                    np.random.default_rng(entry["sample_seed"]).choice(8, 6, replace=False)
                )
                np.testing.assert_array_equal(ids, expected_ids)
                np.testing.assert_array_equal(np.load(path / "vectors.npy"), vectors[ids])
                sampled_payloads = [
                    json.loads(row) for row in (path / "payloads.jsonl").read_text().splitlines()
                ]
                self.assertEqual(sampled_payloads, [payloads[i] for i in ids])
                rows = [json.loads(row) for row in (path / "tests.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), len(queries))
                scores = vectors[ids, 0] / np.linalg.norm(vectors[ids], axis=1)
                for index, row in enumerate(rows):
                    self.assertEqual(row["query"], queries[index]["query"])
                    self.assertEqual(row.get("conditions"), queries[index].get("conditions"))
                    eligible = (
                        np.arange(6)
                        if index == 0
                        else np.flatnonzero(ids % 2 == 0)
                        if index == 1
                        else np.array([], dtype=int)
                    )
                    expected = eligible[np.argsort(-scores[eligible])[:3]]
                    self.assertEqual(row["closest_ids"], expected.tolist())
                    np.testing.assert_allclose(row["closest_scores"], scores[expected], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
