"""Stratified or uniform MiniDBs with exact cosine and geo-radius ground truth."""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import faiss
import numpy as np

from eigen.geo import _build_geo_columns, _condition_mask
from eigen.timing import ConstructionTimer

VECTORS_FILE = "vectors.npy"
PAYLOADS_FILE = "payloads.jsonl"
TESTS_FILE = "tests.jsonl"
SAMPLED_INDICES_FILE = "sampled_indices.npy"
METADATA_FILE = "mini_dataset_metadata.json"


def _as_float32(data: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(data.astype(np.float32, copy=False))


def _validate_input(input_dir: Path) -> Tuple[Path, Optional[Path], Path]:
    vectors_path = input_dir / VECTORS_FILE
    payloads_path = input_dir / PAYLOADS_FILE
    tests_path = input_dir / TESTS_FILE

    if not vectors_path.is_file():
        raise FileNotFoundError(f"Missing {vectors_path}")
    if not tests_path.is_file():
        raise FileNotFoundError(f"Missing {tests_path}")

    return vectors_path, payloads_path if payloads_path.is_file() else None, tests_path


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    known_files = [
        VECTORS_FILE,
        PAYLOADS_FILE,
        TESTS_FILE,
        SAMPLED_INDICES_FILE,
        METADATA_FILE,
    ]
    existing = [output_dir / name for name in known_files if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output files already exist in {output_dir}: {names}. "
            "Pass --overwrite to replace them."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def _load_sampled_payloads(
    payloads_path: Path,
    sampled_indices: np.ndarray,
    expected_count: int,
) -> List[dict]:
    payloads: List[dict] = []
    sample_position = 0
    line_count = 0

    with payloads_path.open("r", encoding="utf-8") as source:
        for original_id, line in enumerate(source):
            line_count = original_id + 1
            if sample_position >= len(sampled_indices):
                continue
            if original_id == int(sampled_indices[sample_position]):
                payloads.append(json.loads(line))
                sample_position += 1

    if line_count != expected_count:
        raise ValueError(
            f"{payloads_path} contains {line_count:,} payloads, but vectors.npy "
            f"contains {expected_count:,} vectors."
        )
    if len(payloads) != len(sampled_indices):
        raise ValueError("Failed to load every payload selected by LSH sampling.")

    return payloads


def _write_payloads(path: Path, payloads: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for payload in payloads:
            output.write(json.dumps(payload, separators=(",", ":")))
            output.write("\n")


def _top_cosine_matches(
    normalized_vectors: np.ndarray,
    normalized_query: np.ndarray,
    candidate_ids: np.ndarray,
    top_k: int,
) -> Tuple[List[int], List[float]]:
    if candidate_ids.size == 0:
        return [], []

    scores = normalized_vectors[candidate_ids] @ normalized_query
    result_count = min(top_k, scores.size)
    if result_count < scores.size:
        local_ids = np.argpartition(scores, -result_count)[-result_count:]
        local_ids = local_ids[np.argsort(scores[local_ids])[::-1]]
    else:
        local_ids = np.argsort(scores)[::-1]

    result_ids = candidate_ids[local_ids]
    result_scores = scores[local_ids]
    return result_ids.astype(int).tolist(), result_scores.astype(float).tolist()


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    normalized = _as_float32(vectors.copy())
    if not np.isfinite(normalized).all() or np.any(np.linalg.norm(normalized, axis=1) == 0):
        raise ValueError("cosine vectors must be finite and nonzero")
    faiss.normalize_L2(normalized)
    return normalized


def _flush_unfiltered_queries(
    pending: List[Tuple[dict, np.ndarray]],
    index: faiss.IndexFlatIP,
    top_k: int,
    output,
) -> int:
    if not pending:
        return 0

    query_vectors = _normalize_vectors(np.stack([query for _, query in pending]))
    scores, ids = index.search(query_vectors, top_k)

    for row_index, (row, _) in enumerate(pending):
        valid = ids[row_index] >= 0
        row["closest_ids"] = ids[row_index][valid].astype(int).tolist()
        row["closest_scores"] = scores[row_index][valid].astype(float).tolist()
        output.write(json.dumps(row, separators=(",", ":")))
        output.write("\n")

    count = len(pending)
    pending.clear()
    return count


def _recompute_tests(
    tests_path: Path,
    output_path: Path,
    mini_vectors: np.ndarray,
    payloads: Optional[List[dict]],
    top_k: int,
    query_batch_size: int,
    max_queries: int,
) -> int:
    if top_k <= 0:
        raise ValueError("top-k must be positive.")
    if query_batch_size <= 0:
        raise ValueError("query-batch-size must be positive.")

    if top_k > len(mini_vectors):
        raise ValueError("MiniDB must contain at least top_k vectors")
    normalized_vectors = _normalize_vectors(mini_vectors)
    index = faiss.IndexFlatIP(normalized_vectors.shape[1])
    index.add(normalized_vectors)
    geo_columns = _build_geo_columns(payloads or [])
    pending: List[Tuple[dict, np.ndarray]] = []
    written = 0

    with (
        tests_path.open("r", encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as output,
    ):
        for source_index, line in enumerate(source):
            if max_queries > 0 and source_index >= max_queries:
                break

            row = json.loads(line)
            query = _as_float32(np.asarray(row["query"], dtype=np.float32))
            if query.ndim != 1 or query.shape[0] != normalized_vectors.shape[1]:
                raise ValueError(
                    f"Query {source_index} has shape {query.shape}; expected "
                    f"({normalized_vectors.shape[1]},)."
                )

            conditions = row.get("conditions")
            if not conditions:
                pending.append((row, query))
                if len(pending) >= query_batch_size:
                    written += _flush_unfiltered_queries(pending, index, top_k, output)
            else:
                written += _flush_unfiltered_queries(pending, index, top_k, output)
                if payloads is None:
                    raise ValueError(
                        "tests.jsonl contains filtered queries, but payloads.jsonl is missing."
                    )

                query_norm = float(np.linalg.norm(query))
                if not np.isfinite(query_norm) or query_norm == 0.0:
                    raise ValueError(f"Query {source_index} has an invalid L2 norm.")
                query /= query_norm
                mask = _condition_mask(conditions, geo_columns, len(mini_vectors))
                candidate_ids = np.flatnonzero(mask)
                ids, scores = _top_cosine_matches(normalized_vectors, query, candidate_ids, top_k)
                row["closest_ids"] = ids
                row["closest_scores"] = scores
                output.write(json.dumps(row, separators=(",", ":")))
                output.write("\n")
                written += 1

            if written and written % 100 == 0:
                print(f"Recomputed queries: {written:,}", end="\r")

        written += _flush_unfiltered_queries(pending, index, top_k, output)

    print(f"Recomputed queries: {written:,}")
    return written


def build_geo_minidbs(
    source: Path,
    output_dir: Path,
    *,
    sample_ratio=0.1,
    sampling_method="locality-stratified",
    sample_seeds=(7630, 7631, 7632),
    bucket_seed=42,
    num_hash_bits=12,
    top_k=100,
    query_batch_size=128,
    overwrite=False,
):
    timer = ConstructionTimer()
    from eigen.minidb import (
        ALLOCATION_VERSION,
        proportional_allocation,
        sampling_partition,
        target_size,
    )
    from eigen.utils import atomic_write_json, dataset_sha256

    source, output_dir = source.resolve(), output_dir.resolve()
    if source == output_dir or source in output_dir.parents:
        raise ValueError("Geo MiniDB output must be outside the source dataset directory")
    if not sample_seeds or len(set(sample_seeds)) != len(sample_seeds):
        raise ValueError("provide distinct sampling seeds")
    vectors_path, payloads_path, tests_path = _validate_input(source)
    vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or not np.issubdtype(vectors.dtype, np.floating):
        raise ValueError("vectors.npy must be a floating-point matrix")
    n = target_size(len(vectors), sample_ratio)
    if not 0 < top_k <= n:
        raise ValueError("MiniDB must contain at least top_k vectors")
    paths = [output_dir / f"mini-{i:02d}" for i in range(len(sample_seeds))]
    if not overwrite and any(p.exists() for p in [*paths, output_dir / "manifest.json"]):
        raise FileExistsError("output exists; choose a new output directory")
    timer.mark("input_and_validation")
    buckets = sampling_partition(
        vectors,
        n,
        sampling_method=sampling_method,
        num_hash_bits=num_hash_bits,
        bucket_seed=bucket_seed,
        metric="cosine",
    )
    timer.mark("bucketization")
    output_dir.mkdir(parents=True, exist_ok=True)
    if sampling_method == "locality-stratified":
        np.save(output_dir / "hash_planes.npy", buckets.planes, allow_pickle=False)
    views = []
    timer.mark("data_write")
    for path, seed in zip(paths, sample_seeds, strict=True):
        _prepare_output(path, overwrite)
        indices = buckets.sample(n, int(seed))
        mini = vectors[indices]  # Preserve original vector values and dtype.
        timer.mark("sampling")
        np.save(path / VECTORS_FILE, mini, allow_pickle=False)
        np.save(path / SAMPLED_INDICES_FILE, indices, allow_pickle=False)
        payloads = None
        if payloads_path is not None:
            payloads = _load_sampled_payloads(payloads_path, indices, len(vectors))
            _write_payloads(path / PAYLOADS_FILE, payloads)
        timer.mark("data_and_payload_io")
        query_count = _recompute_tests(
            tests_path, path / TESTS_FILE, mini, payloads, top_k, query_batch_size, 0
        )
        timer.mark("ground_truth_and_query_write")
        views.append(
            {
                "id": path.name,
                "path": path.name,
                "size": n,
                "sample_seed": int(seed),
                "queries": query_count,
                "sha256": dataset_sha256(path),
            }
        )
        timer.mark("checksums")
    manifest = {
        "schema_version": 1,
        "method": "uniform-without-replacement"
        if sampling_method == "uniform"
        else "shared-hyperplane-stratified",
        "sampling_method": sampling_method,
        "allocation_version": ALLOCATION_VERSION,
        "partition": buckets.partition_metadata,
        "format": "geo",
        "source": str(source),
        "source_sha256": dataset_sha256(source),
        "source_size": len(vectors),
        "dimension": vectors.shape[1],
        "metric": "cosine",
        "top_k": top_k,
        "queries": query_count,
        "bucket_seed": bucket_seed,
        "bucket_fingerprint": buckets.fingerprint,
        "requested_bits": buckets.requested_bits,
        "effective_bits": buckets.effective_bits,
        "nonempty_buckets": len(buckets.counts),
        "target_size": n,
        "allocation": proportional_allocation(buckets.counts, n).tolist(),
        "payload_schema": {key: "geo" for key in _build_geo_columns(payloads or [])},
        "minidbs": views,
    }
    manifest["construction_timing"] = timer.finish()
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument(
        "--sampling-method",
        choices=["locality-stratified", "uniform"],
        default="locality-stratified",
    )
    parser.add_argument("--num-minidbs", type=int, default=3)
    parser.add_argument("--bucket-seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=7630)
    parser.add_argument("--num-hash-bits", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.num_minidbs < 1:
        parser.error("--num-minidbs must be positive")
    result = build_geo_minidbs(
        args.input,
        args.output_dir,
        sample_ratio=args.sample_ratio,
        sampling_method=args.sampling_method,
        bucket_seed=args.bucket_seed,
        sample_seeds=range(args.sample_seed, args.sample_seed + args.num_minidbs),
        num_hash_bits=args.num_hash_bits,
        top_k=args.top_k,
        query_batch_size=args.query_batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
