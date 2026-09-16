from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from mutune.geo_minidb import build_geo_minidbs
from mutune.minidb import (
    build_hdf5_minidbs,
    exact_ground_truth,
    hash_bucketize,
    proportional_allocation,
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


if __name__ == "__main__":
    unittest.main()
