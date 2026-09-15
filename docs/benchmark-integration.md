# Benchmark integration contract

The runner executes `run.py` in a private source copy for every measurement.
It does not import external benchmark code into the tuner or write into the
configured checkout. The following upstream structure is required:

```text
run.py
benchmark/
dataset_reader/
engine/clients/{milvus,qdrant,pgvector}/
engine/base_client/search.py
datasets/datasets.json
```

The CLI must accept the flags assembled in
`src/mutune/runners/vectordb_benchmark.py:_command`: engine, dataset, host,
experiment and internally managed reuse flags. Manual `skip_upload`,
`skip_configure` and `skip_search` settings are rejected, including when
`state_reuse` is false. Use the validated pgvector state-reuse path where
appropriate. One search result must be emitted for the
requested experiment/dataset with `params.experiment`, `params.engine`,
`params.dataset` and finite `results.rps`, `results.mean_precisions`. A fresh
build must emit an upload result with `results.total_time`. Search and upload
identity fields are checked before accepting measurements.

Local data are registered only in the private workspace using an explicit
`runner.settings.dataset_path`. HDF5 uses type `h5`; directories use type
`tar` with a supplied payload schema where applicable. This matches the
upstream [dataset reader dispatch](https://raw.githubusercontent.com/qdrant/vector-db-benchmark/master/benchmark/dataset.py).
No entry needs to be added to the external checkout. A symlink is used where
available; Windows without symlink support gets a private copy.

The Qdrant configurator must forward native collection settings and search
settings, including the quantization object. Its upload stage must wait for
index optimization to finish without overwriting tuned optimizer settings.
The current public [configurator](https://raw.githubusercontent.com/qdrant/vector-db-benchmark/master/engine/clients/qdrant/configure.py)
and [uploader](https://raw.githubusercontent.com/qdrant/vector-db-benchmark/master/engine/clients/qdrant/upload.py)
provide this shape for nonzero `max_optimization_threads`; the profile uses
positive values for this reason. Query exactness is controlled by `config.exact`,
not by calling HNSW a separate exact index. Qdrant 1.16 client fields are defined
in its [versioned model schema](https://raw.githubusercontent.com/qdrant/qdrant-client/v1.16.0/qdrant_client/http/models/models.py).

The Milvus adapter must consume `upload_params.index_type`,
`upload_params.index_params`, and `search_params[0].config` for all seven
families. AUTOINDEX must genuinely be supported by the server distribution;
an unsupported family must produce a failed measurement, not a different
index under the same label. A checked AST compatibility edit removes duplicate
`port` keywords from the public configure/upload/search connection calls in
each private snapshot, preserving the normalized port and other parameters.
The reviewed public source is commit
`e8299454a07d9c429cd1cce9ab610fea68205e44`; actual database compatibility still
requires a smoke run with the pinned server and SDK.

Every private snapshot also replaces the checked `BaseSearcher._search_one`
method body while retaining upstream classes, method signatures and decorators.
Recall is `|returned IDs intersect exact top-K IDs| / |exact top-K IDs|`.
Filtered queries may have fewer than K eligible vectors. Empty exact filtered
answers score 1 only if the database returns no IDs, otherwise 0. Missing ground
truth, duplicate ground-truth IDs and short unfiltered ground truth are errors.
Upstream `mean_precisions` is interpreted as mean Recall under this explicit
contract, recorded in the runner manifest. Raw and previously corrected runs
must not be combined under the same metric definition.

Packaged pgvector overlays provide HNSW, IVFFlat and explicit exact search.
They require the upstream base classes and `get_db_config`, `Record` and `Query`
interfaces. Overlay targets must already exist as regular files; incompatible
checkouts fail closed. The overlays use a logged table, binary COPY, explicit
index construction and per-connection search parameters. Geo filtering is not
supported by the bundled pgvector overlay; the profile advertises no filters.

`build_total_time_s` is the benchmark uploader's total time, including upload
and post-upload work; the timing boundary follows the upstream
[BaseUploader](https://raw.githubusercontent.com/qdrant/vector-db-benchmark/master/engine/base_client/upload.py).
It excludes dataset hashing/staging, service lifecycle and LLM calls. End-to-end
stage wall times are separate. These times are diagnostic/cost records, not
objectives in the default recall-constrained QPS optimization. In particular,
upload-plus-build time is not the pure index construction time used as an
example in paper Section 5.2. Memory usage is not inferred from file size or
LLM output; an explicit extension with a memory objective requires a runner
that actually measures it.

Run `mutune benchmark-fingerprint PATH` to hash copied upstream source.
Generated study configs set `expected_source_sha256`; the runner refuses a
different source digest. MiniDB manifests similarly bind datasets by content
hash. These hashes do not pin Python distributions, database images or drivers:
save the benchmark commit, `pip freeze`, image digests and server versions with
each published experiment. Do not edit a benchmark checkout or dataset during
a running study.

PostgreSQL GUCs are read back after restart. The `milvus_yaml` provider replaces
seven settings in the complete bundled
[Milvus 2.3.1 defaults](https://github.com/milvus-io/milvus/blob/v2.3.1/configs/milvus.yaml)
and mounts the result as `/milvus/configs/milvus.yaml`, the file that version
loads by default. JSON serialization is used as a valid YAML subset.
The old `milvus_user_yaml` provider name remains an alias with the corrected
mount path. `mounted-milvus.yaml` and `milvus-config-verification.json` record
mount verification explicitly; they do not assert that every component's
runtime values were read back. Confirm those effects in the actual experimental
environment before making parameter-level performance claims. The bundled
defaults target 2.3.1; other server versions require corresponding defaults and
integration validation. The converted resource retains upstream attribution;
its Apache 2.0 license is packaged as `mutune/resources/MILVUS-LICENSE`.
