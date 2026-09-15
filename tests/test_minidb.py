from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from mutune.minidb import (
    build_hdf5_minidbs,
    exact_ground_truth,
    hash_bucketize,
    proportional_allocation,
)


class MiniDBTests(unittest.TestCase):
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
            manifest = build_hdf5_minidbs(source, root / "views", sample_ratio=0.2, top_k=3)
            self.assertEqual(len(manifest["minidbs"]), 3)
            self.assertEqual(sum(manifest["allocation"]), 10)
            for entry in manifest["minidbs"]:
                path = root / "views" / entry["path"]
                ids = np.load(path.with_suffix(".indices.npy"))
                with h5py.File(path, "r") as f:
                    np.testing.assert_array_equal(f["train"][:], original[ids])
                    np.testing.assert_array_equal(f["test"][:], queries)
                    self.assertLess(f["neighbors"][:].max(), 10)
                    self.assertEqual(f.attrs["bucket_fingerprint"], manifest["bucket_fingerprint"])
            with self.assertRaises(FileExistsError):
                build_hdf5_minidbs(source, root / "views", top_k=3)


if __name__ == "__main__":
    unittest.main()
