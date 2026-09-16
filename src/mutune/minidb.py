"""Shared-bucket MiniDB construction (paper Section 4.1 / Algorithm 1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mutune.timing import ConstructionTimer
from mutune.utils import atomic_write_json, dataset_sha256


def proportional_allocation(counts: np.ndarray, size: int) -> np.ndarray:
    """Bounded largest-remainder allocation: each bucket gets >=1, sum is n.

    Start with bounded floors of proportional quotas. Add largest residuals,
    or remove greatest overallocations when minimum-one rounding exceeds n.
    Stable bucket order breaks ties; capacities and lower bounds are respected.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1 or not len(counts) or np.any(counts <= 0):
        raise ValueError("counts must contain positive nonempty bucket sizes")
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise ValueError("target size must be an integer")
    if not len(counts) <= size <= int(counts.sum()):
        raise ValueError("require number of buckets <= target size <= N")
    quotas = counts.astype(np.float64) * (size / int(counts.sum()))
    allocated = np.minimum(counts, np.maximum(1, np.floor(quotas).astype(np.int64)))
    while int(allocated.sum()) != size:
        deficit = size - int(allocated.sum())
        eligible = np.flatnonzero(allocated < counts if deficit > 0 else allocated > 1)
        residual = quotas - allocated if deficit > 0 else allocated - quotas
        order = eligible[np.argsort(-residual[eligible], kind="stable")]
        chosen = order[: abs(deficit)]
        if not len(chosen):
            raise ValueError("no capacity to satisfy the bounded allocation")
        allocated[chosen] += 1 if deficit > 0 else -1
    return allocated


@dataclass(frozen=True)
class HashBuckets:
    planes: np.ndarray
    codes: np.ndarray
    order: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    requested_bits: int
    effective_bits: int
    bucket_seed: int
    metric: str
    offsets: np.ndarray
    l2_bucket_width: float | None

    @property
    def family(self) -> str:
        return "euclidean-p-stable" if self.metric == "l2" else "angular-hyperplane"

    @property
    def partition_metadata(self) -> dict:
        return {
            "family": self.family,
            "metric": self.metric,
            "requested_hash_functions": self.requested_bits,
            "effective_hash_functions": self.effective_bits,
            "projections_file": "hash_planes.npy",
            "offsets_file": "hash_offsets.npy" if self.metric == "l2" else None,
            "l2_bucket_width": self.l2_bucket_width,
        }

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.family.encode("ascii"))
        digest.update(self.metric.encode("ascii"))
        digest.update(np.asarray([self.l2_bucket_width or 0], dtype="<f8").tobytes())
        digest.update(np.asarray(self.planes, dtype="<f8").tobytes())
        digest.update(np.asarray(self.offsets, dtype="<f8").tobytes())
        digest.update(np.asarray(self.codes, dtype="<u8").tobytes())
        return digest.hexdigest()

    def sample(self, size: int, seed: int) -> np.ndarray:
        allocation = proportional_allocation(self.counts, size)
        rng = np.random.default_rng(seed)
        parts = [
            rng.choice(self.order[start : start + count], size=int(take), replace=False)
            for start, count, take in zip(self.starts, self.counts, allocation, strict=True)
        ]
        return np.sort(np.concatenate(parts).astype(np.int64))


def hash_bucketize(
    vectors: np.ndarray,
    size: int,
    *,
    num_hash_bits: int = 12,
    bucket_seed: int = 42,
    chunk_size: int = 65536,
    metric: str = "cosine",
    l2_bucket_width: float = 1.0,
) -> HashBuckets:
    """Select the longest shared LSH prefix with at most ``size`` buckets.

    Cosine/angular uses signs of Gaussian projections. Euclidean L2 uses
    h(x) = floor((a.x + b) / w), with a ~ N(0, I) and b ~ Uniform[0, w),
    the p-stable LSH family of Datar et al. (SoCG 2004), p=2:
    https://doi.org/10.1145/997817.997857. Width w is in the input's units;
    vectors are never normalized for this partition. Legacy dot workloads
    retain angular partitioning (not a general inner-product LSH guarantee).

    The seed defines a fixed ordered sequence, independent of the requested
    prefix length. Each added function only refines existing buckets, so the
    first excessive prefix ends the search. The empty prefix is one bucket.
    ``num_hash_bits`` names the prefix length for API compatibility; L2 hash
    values are signed grid coordinates, not bits.
    """
    if vectors.ndim != 2 or not len(vectors) or vectors.shape[1] < 1:
        raise ValueError("vectors must be a non-empty N by d array")
    if not 0 < size <= len(vectors) or not 1 <= num_hash_bits <= 63 or chunk_size < 1:
        raise ValueError("require 0 < n <= N, 1 <= hash bits <= 63, positive chunks")
    metric = {"angular": "cosine", "euclidean": "l2"}.get(metric, metric)
    if metric not in {"cosine", "l2", "dot"}:
        raise ValueError("metric must be l2/euclidean, cosine/angular, or dot")
    if not math.isfinite(l2_bucket_width) or l2_bucket_width <= 0:
        raise ValueError("l2_bucket_width must be finite and positive")
    projection_seed, offset_seed = np.random.SeedSequence(bucket_seed).spawn(2)
    planes = (
        np.random.default_rng(projection_seed).standard_normal((num_hash_bits, vectors.shape[1])).T
    )
    offsets = np.empty(0, dtype=np.float64)
    if metric == "l2":
        offsets = np.random.default_rng(offset_seed).uniform(0, l2_bucket_width, num_hash_bits)
        codes = np.zeros(len(vectors), dtype=np.uint64)
        # Losslessly compress (previous-prefix ID, next grid coordinate).
        # This avoids fixed-width bit packing, collisions and N-by-prefix storage.
        pairs = np.empty(len(vectors), dtype=[("prefix", "<u8"), ("cell", "<i8")])
        bits = 0
        for column in range(num_hash_bits):
            pairs["prefix"] = codes
            for start in range(0, len(vectors), chunk_size):
                block = np.asarray(vectors[start : start + chunk_size])
                if not np.isfinite(block).all():
                    raise ValueError("vectors contain NaN or infinity")
                cells = np.floor((block @ planes[:, column] + offsets[column]) / l2_bucket_width)
                if (
                    not np.isfinite(cells).all()
                    or np.any(cells < -(2**63))
                    or np.any(cells >= 2**63)
                ):
                    raise ValueError("L2 grid coordinates exceed int64; increase l2_bucket_width")
                pairs["cell"][start : start + len(block)] = cells.astype(np.int64)
            unique, inverse = np.unique(pairs, return_inverse=True)
            if len(unique) > size:
                break
            codes = inverse.astype(np.uint64)
            bits += 1
    else:
        codes = np.empty(len(vectors), dtype=np.uint64)
        powers = np.left_shift(np.uint64(1), np.arange(num_hash_bits - 1, -1, -1, dtype=np.uint64))
        for start in range(0, len(vectors), chunk_size):
            block = np.asarray(vectors[start : start + chunk_size])
            if not np.isfinite(block).all():
                raise ValueError("vectors contain NaN or infinity")
            codes[start : start + len(block)] = (block @ planes >= 0).astype(np.uint64) @ powers
        bits = num_hash_bits
        while len(np.unique(codes)) > size:
            codes >>= np.uint64(1)
            bits -= 1
    order = np.argsort(codes, kind="stable")
    _, starts, counts = np.unique(codes[order], return_index=True, return_counts=True)
    return HashBuckets(
        planes[:, :bits],
        codes,
        order,
        starts,
        counts,
        num_hash_bits,
        bits,
        bucket_seed,
        metric,
        offsets[:bits],
        l2_bucket_width if metric == "l2" else None,
    )


def exact_ground_truth(
    train: np.ndarray, queries: np.ndarray, metric: str, top_k: int, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    import faiss

    if train.ndim != 2 or queries.ndim != 2 or train.shape[1] != queries.shape[1]:
        raise ValueError("train and queries must have matching dimensions")
    if not 0 < top_k <= len(train) or batch_size < 1:
        raise ValueError("require 0 < top_k <= MiniDB size and positive query batch size")
    x = np.array(train, dtype=np.float32, order="C", copy=True)
    q = np.array(queries, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError("non-finite vectors or queries")
    metric = {"angular": "cosine", "euclidean": "l2"}.get(metric, metric)
    if metric == "cosine":
        if np.any(np.linalg.norm(x, axis=1) == 0) or np.any(np.linalg.norm(q, axis=1) == 0):
            raise ValueError("cosine distance is undefined for zero vectors")
        faiss.normalize_L2(x)
        faiss.normalize_L2(q)
    if metric not in {"cosine", "l2", "dot"}:
        raise ValueError("metric must be l2/euclidean, cosine/angular, or dot")
    index = faiss.IndexFlatL2(x.shape[1]) if metric == "l2" else faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    distances = np.empty((len(q), top_k), dtype=np.float32)
    neighbors = np.empty((len(q), top_k), dtype=np.int64)
    for start in range(0, len(q), batch_size):
        scores, ids = index.search(q[start : start + batch_size], top_k)
        d = (
            np.sqrt(np.maximum(scores, 0))
            if metric == "l2"
            else (1 - scores if metric == "cosine" else -scores)
        )
        distances[start : start + len(scores)] = d
        neighbors[start : start + len(scores)] = ids
    return distances, neighbors


def target_size(count: int, sample_ratio: float, size: int | None = None) -> int:
    if not math.isfinite(sample_ratio) or not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio must be in (0, 1]")
    result = int(count * sample_ratio) if size is None else size
    if not 0 < result <= count:
        raise ValueError("requested sample contains no vectors or exceeds N")
    return result


def build_hdf5_minidbs(
    source: Path,
    output_dir: Path,
    *,
    sample_ratio: float = 0.1,
    size: int | None = None,
    sample_seeds: Sequence[int] = (7630, 7631, 7632),
    bucket_seed: int = 42,
    num_hash_bits: int = 12,
    l2_bucket_width: float = 1.0,
    top_k: int = 100,
    metric: str = "l2",
    overwrite: bool = False,
    query_batch_size: int = 256,
) -> dict:
    timer = ConstructionTimer()
    import h5py

    source, output_dir = source.resolve(), output_dir.resolve()
    if not sample_seeds or len(set(sample_seeds)) != len(sample_seeds):
        raise ValueError("provide distinct sampling seeds for the MiniDB views")
    paths = [output_dir / f"mini-{i:02d}.hdf5" for i in range(len(sample_seeds))]
    reserved = paths + [
        output_dir / "manifest.json",
        output_dir / "hash_planes.npy",
        output_dir / "hash_offsets.npy",
    ]
    reserved += [p.with_suffix(".indices.npy") for p in paths]
    if source in reserved:
        raise ValueError("output must not overwrite the source dataset")
    if not overwrite and any(p.exists() for p in reserved):
        raise FileExistsError("MiniDB output exists; use a new directory or --overwrite")
    with h5py.File(source, "r") as f:
        train, queries = np.asarray(f["train"]), np.asarray(f["test"])
    if not len(queries) or not np.issubdtype(train.dtype, np.floating):
        raise ValueError("require floating-point train vectors and nonempty queries")
    metric = {"angular": "cosine", "euclidean": "l2"}.get(metric, metric)
    n = target_size(len(train), sample_ratio, size)
    if not 0 < top_k <= n:
        raise ValueError("MiniDB must contain at least top_k vectors")
    timer.mark("input_and_validation")
    buckets = hash_bucketize(
        train,
        n,
        num_hash_bits=num_hash_bits,
        bucket_seed=bucket_seed,
        metric=metric,
        l2_bucket_width=l2_bucket_width,
    )
    timer.mark("bucketization")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "hash_planes.npy", buckets.planes, allow_pickle=False)
    if metric == "l2":
        np.save(output_dir / "hash_offsets.npy", buckets.offsets, allow_pickle=False)
    views = []
    timer.mark("data_write")
    for index, (path, seed) in enumerate(zip(paths, sample_seeds, strict=True)):
        indices = buckets.sample(n, int(seed))
        mini = train[indices]
        timer.mark("sampling")
        distances, neighbors = exact_ground_truth(mini, queries, metric, top_k, query_batch_size)
        timer.mark("ground_truth")
        temporary = path.with_suffix(".partial.hdf5")
        with h5py.File(temporary, "w") as f:
            f.attrs.update(
                {
                    "type": "dense",
                    "distance": metric,
                    "dimension": train.shape[1],
                    "source": str(source),
                    "bucket_seed": bucket_seed,
                    "sample_seed": seed,
                    "bucket_fingerprint": buckets.fingerprint,
                }
            )
            for name, data in (
                ("train", mini),
                ("test", queries),
                ("distances", distances),
                ("neighbors", neighbors),
            ):
                f.create_dataset(name, data=data)
        os.replace(temporary, path)
        np.save(path.with_suffix(".indices.npy"), indices, allow_pickle=False)
        timer.mark("data_write")
        views.append(
            {
                "id": f"mini-{index:02d}",
                "path": path.name,
                "sample_seed": int(seed),
                "size": n,
                "sha256": dataset_sha256(path),
                "indices": path.with_suffix(".indices.npy").name,
            }
        )
        timer.mark("checksums")
    manifest = {
        "schema_version": 1,
        "method": "shared-p-stable-stratified"
        if metric == "l2"
        else "shared-hyperplane-stratified",
        "partition": buckets.partition_metadata,
        "source": str(source),
        "source_sha256": dataset_sha256(source),
        "source_size": len(train),
        "dimension": train.shape[1],
        "queries": len(queries),
        "metric": metric,
        "top_k": top_k,
        "bucket_seed": bucket_seed,
        "bucket_fingerprint": buckets.fingerprint,
        "requested_bits": num_hash_bits,
        "effective_bits": buckets.effective_bits,
        "nonempty_buckets": len(buckets.counts),
        "target_size": n,
        "allocation": proportional_allocation(buckets.counts, n).tolist(),
        "minidbs": views,
    }
    manifest["construction_timing"] = timer.finish()
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build MiniDBs using one shared LSH partition")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--size", type=int)
    parser.add_argument("--num-minidbs", type=int, default=3)
    parser.add_argument("--bucket-seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=7630)
    parser.add_argument("--num-hash-bits", type=int, default=12)
    parser.add_argument(
        "--l2-bucket-width",
        type=float,
        default=1.0,
        help="Euclidean p-stable grid width in original vector units (default: 1.0)",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--metric", choices=["l2", "euclidean", "cosine", "angular", "dot"], default="l2"
    )
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.num_minidbs < 1:
        parser.error("--num-minidbs must be positive")
    manifest = build_hdf5_minidbs(
        args.input,
        args.output_dir,
        sample_ratio=args.sample_ratio,
        size=args.size,
        sample_seeds=range(args.sample_seed, args.sample_seed + args.num_minidbs),
        bucket_seed=args.bucket_seed,
        num_hash_bits=args.num_hash_bits,
        l2_bucket_width=args.l2_bucket_width,
        top_k=args.top_k,
        metric=args.metric,
        overwrite=args.overwrite,
        query_batch_size=args.query_batch_size,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
