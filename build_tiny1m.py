"""Optional contiguous Tiny subset conversion; not the paper MiniDB sampler."""

import argparse
import os
import time
from pathlib import Path
from typing import BinaryIO, Iterator, Tuple

import faiss
import h5py
import numpy as np

DEFAULT_OUTPUT = Path("datasets/tiny-1m-384-euclidean/tiny-1m-384-euclidean.hdf5")


def open_fvecs(path: Path, dimension: int) -> Tuple[np.memmap, int]:
    record_bytes = 4 * (dimension + 1)
    file_size = path.stat().st_size
    if file_size % record_bytes != 0:
        raise ValueError(
            f"{path} has {file_size} bytes, which is not divisible by the "
            f"{record_bytes}-byte record size for {dimension}D fvecs"
        )

    count = file_size // record_bytes
    records = np.memmap(
        path,
        dtype="<i4",
        mode="r",
        shape=(count, dimension + 1),
    )
    return records, count


def read_fvecs_batch(
    records: np.memmap,
    start: int,
    stop: int,
    dimension: int,
) -> np.ndarray:
    headers = records[start:stop, 0]
    invalid = np.flatnonzero(headers != dimension)
    if invalid.size:
        record_id = start + int(invalid[0])
        raise ValueError(
            f"Invalid fvecs dimension at record {record_id}: "
            f"expected {dimension}, found {int(headers[invalid[0]])}"
        )

    payload_bits = np.ascontiguousarray(records[start:stop, 1:])
    vectors = payload_bits.view("<f4")
    if not np.isfinite(vectors).all():
        raise ValueError(f"Non-finite vector value found in records [{start}, {stop})")
    return vectors


def selected_ranges(
    start_id: int,
    sample_size: int,
    total_count: int,
    chunk_size: int,
) -> Iterator[Tuple[int, int, int]]:
    output_start = 0
    source_start = start_id
    remaining = sample_size

    while remaining:
        available = min(remaining, total_count - source_start)
        consumed = 0
        while consumed < available:
            count = min(chunk_size, available - consumed)
            yield source_start + consumed, source_start + consumed + count, output_start
            consumed += count
            output_start += count
            remaining -= count
        source_start = 0


def write_fvecs_batch(file: BinaryIO, vectors: np.ndarray) -> None:
    records = np.empty((len(vectors), vectors.shape[1] + 1), dtype="<i4")
    records[:, 0] = vectors.shape[1]
    records[:, 1:] = np.ascontiguousarray(vectors, dtype="<f4").view("<i4")
    file.write(records.tobytes())


def write_ivecs_batch(file: BinaryIO, vectors: np.ndarray) -> None:
    records = np.empty((len(vectors), vectors.shape[1] + 1), dtype="<i4")
    records[:, 0] = vectors.shape[1]
    records[:, 1:] = np.ascontiguousarray(vectors, dtype="<i4")
    file.write(records.tobytes())


def ensure_output_available(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def build_tiny1m(
    base_path: Path,
    query_path: Path,
    output_path: Path,
    dimension: int,
    sample_size: int,
    query_count: int,
    top_k: int,
    seed: int,
    start_id: int | None,
    chunk_size: int,
    query_batch_size: int,
    threads: int,
    pgtuner_output_root: Path | None,
    overwrite: bool,
) -> None:
    started_at = time.time()
    base_records, total_count = open_fvecs(base_path, dimension)
    query_records, available_queries = open_fvecs(query_path, dimension)

    if not 0 < sample_size <= total_count:
        raise ValueError(f"sample_size must be in [1, {total_count}], got {sample_size}")
    if not 0 < query_count <= available_queries:
        raise ValueError(f"query_count must be in [1, {available_queries}], got {query_count}")
    if not 0 < top_k <= sample_size:
        raise ValueError(f"top_k must be in [1, {sample_size}], got {top_k}")
    if chunk_size <= 0 or query_batch_size <= 0 or threads <= 0:
        raise ValueError("chunk_size, query_batch_size, and threads must be positive")

    if start_id is None:
        # Match np.random.seed(seed); np.random.randint(...) used by PGTuner.
        start_id = int(np.random.RandomState(seed).randint(0, total_count))
    if not 0 <= start_id < total_count:
        raise ValueError(f"start_id must be in [0, {total_count}), got {start_id}")

    ensure_output_available(output_path, overwrite)
    partial_output = output_path.with_suffix(output_path.suffix + ".partial")
    if partial_output.exists():
        if overwrite:
            partial_output.unlink()
        else:
            raise FileExistsError(
                f"Partial output already exists: {partial_output}. Remove it or use --overwrite."
            )

    pgtuner_base_path = None
    pgtuner_query_path = None
    pgtuner_gt_path = None
    pgtuner_base_partial = None
    if pgtuner_output_root is not None:
        pgtuner_base_path = pgtuner_output_root / "Base/tiny/1_1_384.fvecs"
        pgtuner_query_path = pgtuner_output_root / "Query/tiny/384.fvecs"
        pgtuner_gt_path = pgtuner_output_root / "GroundTruth/tiny/1_1_384.ivecs"
        for path in (pgtuner_base_path, pgtuner_query_path, pgtuner_gt_path):
            ensure_output_available(path, overwrite)
        pgtuner_base_partial = pgtuner_base_path.with_suffix(".fvecs.partial")
        if pgtuner_base_partial.exists():
            if overwrite:
                pgtuner_base_partial.unlink()
            else:
                raise FileExistsError(f"Partial output already exists: {pgtuner_base_partial}")

    print(
        f"Tiny5M base: {total_count:,} vectors x {dimension}D\n"
        f"Tiny1M subset: start_id={start_id:,}, size={sample_size:,}, seed={seed}\n"
        f"Queries: {query_count:,}, ground truth: Top-{top_k}",
        flush=True,
    )

    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatL2(dimension)
    queries = read_fvecs_batch(query_records, 0, query_count, dimension)

    base_file = open(pgtuner_base_partial, "wb") if pgtuner_base_partial else None
    try:
        with h5py.File(partial_output, "w") as output:
            output.attrs["type"] = "dense"
            output.attrs["distance"] = "euclidean"
            output.attrs["dimension"] = dimension
            output.attrs["point_type"] = "float"
            output.attrs["source"] = str(base_path)
            output.attrs["subset_seed"] = seed
            output.attrs["subset_start_id"] = start_id
            output.attrs["subset_size"] = sample_size

            train_ds = output.create_dataset(
                "train",
                shape=(sample_size, dimension),
                dtype="f4",
            )
            output.create_dataset("test", data=queries, dtype="f4")
            neighbors_ds = output.create_dataset(
                "neighbors", shape=(query_count, top_k), dtype="i4"
            )
            distances_ds = output.create_dataset(
                "distances", shape=(query_count, top_k), dtype="f4"
            )

            uploaded = 0
            for source_start, source_stop, output_start in selected_ranges(
                start_id, sample_size, total_count, chunk_size
            ):
                vectors = read_fvecs_batch(base_records, source_start, source_stop, dimension)
                output_stop = output_start + len(vectors)
                train_ds[output_start:output_stop] = vectors
                index.add(vectors)
                if base_file is not None:
                    write_fvecs_batch(base_file, vectors)
                uploaded = output_stop
                if uploaded % 100_000 == 0 or uploaded == sample_size:
                    print(f"Loaded {uploaded:,}/{sample_size:,} base vectors", flush=True)

            print("Computing exact L2 ground truth with FAISS...", flush=True)
            for start in range(0, query_count, query_batch_size):
                stop = min(start + query_batch_size, query_count)
                squared_distances, neighbors = index.search(queries[start:stop], top_k)
                neighbors_ds[start:stop] = neighbors.astype("i4", copy=False)
                distances_ds[start:stop] = np.sqrt(np.maximum(squared_distances, 0.0))
                print(f"Processed {stop:,}/{query_count:,} queries", flush=True)
    except Exception:
        partial_output.unlink(missing_ok=True)
        if pgtuner_base_partial is not None:
            pgtuner_base_partial.unlink(missing_ok=True)
        raise
    finally:
        if base_file is not None:
            base_file.close()

    os.replace(partial_output, output_path)

    if pgtuner_output_root is not None:
        os.replace(pgtuner_base_partial, pgtuner_base_path)
        with open(pgtuner_query_path, "wb") as query_file:
            write_fvecs_batch(query_file, queries)
        with h5py.File(output_path, "r") as output:
            neighbors = np.asarray(output["neighbors"])
        with open(pgtuner_gt_path, "wb") as gt_file:
            write_ivecs_batch(gt_file, neighbors)

    elapsed = time.time() - started_at
    print(f"Created {output_path} in {elapsed:.1f} seconds", flush=True)
    if pgtuner_output_root is not None:
        print(f"Created PGTuner files under {pgtuner_output_root}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a reproducible 1M subset from Tiny5M and build an "
            "ANN-Benchmarks-compatible HDF5 dataset with exact L2 ground truth."
        )
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--query-count", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help="First Tiny5M record to extract; default derives it from --seed.",
    )
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--query-batch-size", type=int, default=10)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--pgtuner-output-root",
        type=Path,
        default=None,
        help=(
            "Optionally also write Base/tiny/1_1_384.fvecs, Query/tiny/384.fvecs, "
            "and GroundTruth/tiny/1_1_384.ivecs below this directory."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_tiny1m(
        base_path=args.base.expanduser().resolve(),
        query_path=args.queries.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        dimension=args.dimension,
        sample_size=args.sample_size,
        query_count=args.query_count,
        top_k=args.top_k,
        seed=args.seed,
        start_id=args.start_id,
        chunk_size=args.chunk_size,
        query_batch_size=args.query_batch_size,
        threads=args.threads,
        pgtuner_output_root=(
            args.pgtuner_output_root.expanduser().resolve()
            if args.pgtuner_output_root is not None
            else None
        ),
        overwrite=args.overwrite,
    )
