# Reproducing EIGEN experiments

This guide targets the accompanying [current manuscript](EIGEN_VLDB.pdf), Sections
6.1–6.7. It provides executable preparation and aggregation tools.

## 1. Prepare and record the environment

Use Python 3.11+ for EIGEN and a separate compatible environment for
[vector-db-benchmark](https://github.com/qdrant/vector-db-benchmark). Install from
the repository root, then run the offline checks:

```bash
python -m pip install -e ".[data,dev]"
python scripts/check_static.py
python -m unittest discover -s tests
```

The core dependency is Pydantic; Mini-DB builders additionally use NumPy, h5py and
FAISS. Docker Compose, the database SDKs and benchmark dependencies are external
requirements. Read the [adapter contract](benchmark-integration.md) before
selecting a benchmark checkout. The reviewed source commit in that document is
a compatibility reference, not proof that all engine/SDK combinations have been
tested. Run at least one real upload/index/query smoke evaluation for each
engine/profile before starting a large matrix. Check that every declared family
is actually supported by the server distribution, especially Milvus AUTOINDEX.

Paper versions: Milvus 2.3.1, Qdrant 1.16.0, pgvector 0.6.2. The PostgreSQL 16 base
image and local Compose arrangement are repository choices. The paper uses
16-core 3.60-GHz CPUs and 64 GB RAM per machine, with concurrent Mini-DB workers
on separate identically provisioned machines. Generated files initially use
localhost containers. Review host placement, CPU affinity, memory and storage
isolation; several containers sharing one machine do not reproduce that testbed.
The current Compose runner operates locally and does not automate remote worker
placement. Reproducing the paper's multi-machine arrangement requires remote
orchestration integration and per-machine setup. Merely changing a host or using
the external lifecycle is insufficient: the external lifecycle rejects tuned
system-knob reconfiguration. A single-host run with documented resource isolation
is a new experimental deployment, not the exact original machine arrangement.
The Compose templates delete their own experimental volumes on stage shutdown.

Save actual environment information with each published experiment. These are
commands to run in the corresponding environments, not bundled measurements:

```bash
git rev-parse HEAD
python --version
python -m pip freeze
git -C external/vector-db-benchmark rev-parse HEAD
/absolute/path/to/benchmark-python -m pip freeze
docker version
docker compose version
docker image inspect milvusdb/milvus:v2.3.1
docker image inspect qdrant/qdrant:v1.16.0
```

Record PostgreSQL/pgvector server versions and the locally built image digest as
well. Save OS/kernel, physical machine identifiers, CPU model, memory, storage,
network placement, exact project JSONs, dataset hashes, model identifier/endpoint,
request settings and token rates. Do not put API keys in JSON or snapshots. The
project pins a benchmark *source-content hash* but deliberately does not invent a
verified dependency lock or historical container digest. An exact environment
lock requires the authors' tested environment.

## 2. Obtain and verify the six source workloads

The following specifications and links are transcribed from Table 3 / Section
6.1 of the supplied manuscript; downloads are not performed by this repository.

| Inventory name | Vectors | Queries | Dimensions | Metric | Manuscript source |
| --- | ---: | ---: | ---: | --- | --- |
| `nytimes` | 290,000 | 1,000 | 256 | cosine | [ANN-Benchmarks](https://ann-benchmarks.com/nytimes-256-angular.hdf5) |
| `glove` | 1,183,514 | 10,000 | 100 | cosine | [ANN-Benchmarks](http://ann-benchmarks.com/glove-100-angular.hdf5) |
| `gist` | 1,000,000 | 1,000 | 960 | l2 | [ANN-Benchmarks](http://ann-benchmarks.com/gist-960-euclidean.hdf5) |
| `tiny5m` | 5,000,000 | 1,000 | 384 | l2 | [GQR archive](https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/tiny5m.tar.gz) |
| `msong` | 994,185 | 200 | 420 | l2 | [GQR archive](https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/msong.tar.gz) |
| `geo-radius` | 100,000 | 1,000 | 2,048 | cosine | [Filtered benchmark archive](https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_geo_1m.tgz) |

Dense source files must use the benchmark HDF5 layout (`train`, `test`, exact
`neighbors` and distances where required by the selected reader). The Mini-DB
builder recomputes ground truth for each Mini-DB; it does not repair missing or
wrong *full-database* ground truth. Preserve the query set and document any
conversion, normalization, subsampling, duplicate handling and exact-neighbor
procedure. Tiny5M/Msong archive-to-HDF5 conversion and the exact 100,000-vector
Geo-radius subset are not reconstructed from the paper. In particular, the Geo
download name says `1m` while Table 3 says 100,000: the original subset/query
recipe or author-supplied ready-to-use workload is required for an exact match.
Do not silently choose a subset and label it the original paper dataset.

Geo-radius uses a directory with `vectors.npy`, `payloads.jsonl` and
`tests.jsonl`. Payload rows must correspond to vector rows; tests contain the
query vector, `and`/`or` geo-radius conditions and exact eligible `closest_ids`.

## 3. Build independently sampled Mini-DBs

Example for an L2 source; set the metric and input path for each workload:

```bash
eigen-build-minidbs --input data/tiny5m/source.hdf5 --output-dir data/tiny5m/stratified-m3-r10 --num-minidbs 3 --sample-ratio 0.1 --bucket-seed 42 --sample-seed 7630 --metric l2 --l2-bucket-width 1.0 --top-k 100
eigen-build-geo-minidbs --input data/geo-radius/source --output-dir data/geo-radius/stratified-m3-r10 --num-minidbs 3 --sample-ratio 0.1 --bucket-seed 42 --sample-seed 7630 --top-k 100
```

These calls perform real data processing and exact ground-truth construction,
but do not call an LLM or database. Each manifest records source and Mini-DB
hashes, sampling seeds, partition information and construction timing. Rebuild
older Mini-DBs: the matrix generator requires the current stratified allocation
version `minimum-one-original-population-largest-remainder-v2`, and refuses old
or unknown versions rather than treating old quota allocations as current results.
Never modify the data after the manifest is generated. Record the bucket width, hash
bits and seeds: their CLI defaults are implementation choices where the paper
does not specify exact values. The default query `top_k` in project generation
is 10; explicitly set the value used by the experiment and ensure manifest
ground truth has at least that many neighbors.

For Uniform-Mini, rebuild into a **different** directory using the same input,
size, query set and seeds, adding `--sampling-method uniform` to either builder.
Do not relabel a stratified manifest as uniform. For the Section 6.6.1 sensitivity
study, build independent manifests for M = 1, 2, 3, 4, 5 at ratio 0.10, and ratios
0.01, 0.05, 0.10, 0.20 at M = 3. Use distinct output directories. No manifest is
silently truncated to simulate a different number of Mini-DBs.

## 4. Generate and inspect the experiment matrix

Copy [the inventory template](reproduction-input.example.json) to the repository
root as `reproduction-input.json`. Its paths are relative to that copied inventory.
Fill in the six actual manifest paths and descriptions. Add additional entries
to a dataset's `manifests` list for uniform or sensitivity inputs, for example:

```json
{"label": "uniform-m3-r10", "path": "data/tiny5m/uniform-m3-r10/manifest.json"}
```

Generate the main 6 × 7 matrix (Milvus, one tuning seed, one model):

```bash
python scripts/prepare_reproduction.py --inventory reproduction-input.json --benchmark-repo external/vector-db-benchmark --benchmark-python /absolute/path/to/benchmark-python --output experiments/main --require-paper-datasets
```

The seven default recalls are 0.85, 0.875, 0.90, 0.925, 0.95, 0.975 and 0.99.
Each cell uses one fixed threshold. The defaults are M from the manifest, 20
local evaluations per Mini-DB, `initial_samples: null` (one per conditional
region, seven on Milvus), 12 proposals, batch size 4, stability weight 1 and
shortlist L = 5. Cross-Mini-DB and final measurements are additional physical
evaluations. `--budget-per-minidb`, `--top-l`, `--top-k`, `--search-parallel`,
`--upload-parallel`, `--recalls` and `--seeds` are explicit overrides. Multiple
tuning seeds rerun optimization on the same manifests; independent data-sampling
replicates require separately built and listed manifests.

The script writes project/study JSONs, copied deployment templates,
`run-plan.json` and `COMMANDS.md`. It executes zero experiments. Configs and plan
paths are resolved for the preparation machine; regenerate after moving the
workspace to another machine. Each model and manifest has a separate output
directory. Existing output directories must be empty.

Useful matrix variants:

```bash
# Supported framework ablations; inventory must include uniform manifests.
python scripts/prepare_reproduction.py --inventory tiny5m-input.json --benchmark-repo external/vector-db-benchmark --benchmark-python /absolute/path/to/benchmark-python --output experiments/ablations --recalls 0.95 --methods eigen mean-only uniform-mini
# Direct-Full: 60 here is an explicitly chosen full-database search budget.
python scripts/prepare_reproduction.py --inventory tiny5m-input.json --benchmark-repo external/vector-db-benchmark --benchmark-python /absolute/path/to/benchmark-python --output experiments/direct --recalls 0.95 --methods direct-full --direct-full-budget 60
# Sensitivity: list the independently built M/ratio manifests in the inventory.
python scripts/prepare_reproduction.py --inventory sensitivity-input.json --benchmark-repo external/vector-db-benchmark --benchmark-python /absolute/path/to/benchmark-python --output experiments/sensitivity --recalls 0.95
```

Mean-Only sets stability weight to zero and retains cross-view feasibility checks.
Uniform-Mini retains the rest of the framework. Direct-Full calls CALM's local
optimizer on the original database without Mini-DB construction or transfer.
It requires an explicit budget, since EIGEN's 60 local evaluations do not include
cross-view and shortlist measurements. To reproduce the equal-*total*-physical-
evaluation comparison, first obtain the EIGEN run's actual physical count, then
set the Direct-Full budget accordingly. The paper's equal-aggregate-time variant
needs a validated time-budget stopping policy; this generator does not implement
one and does not claim that a fixed evaluation count is time-equivalent.

Use `--engines qdrant pgvector` for engine variants; unsupported pgvector Geo
cells are explicitly recorded in `skipped`, not simulated. Use `--backbones
backbones.json` with the format in [the backbone template](backbones.example.json)
for model/reasoning sweeps. Each `llm` object follows `LLMConfig`, including
`base_url`, `model`, `api_key_env`, `extra_body` and optional explicit token rates.
The default label follows the manuscript's GPT-5.4 medium setting; endpoint and
model availability require verification in the actual experimental account.
Different providers may require different request fields. Only compatible
endpoints returning the required structured output can be substituted.

## 5. Run cells and export actual measurements

Review the generated JSONs and deploy the chosen isolated testbed first. Set
the API key in the named environment variable. `COMMANDS.md` and `run-plan.json`
contain exact argument arrays for offline validation followed by execution.
For example, select one cell and run the equivalent commands:

```bash
eigen study experiments/main/nytimes/milvus/eigen/stratified-m3-r10/gpt-5_4-medium/r0.95-s42/study.json --validate-only
eigen study experiments/main/nytimes/milvus/eigen/stratified-m3-r10/gpt-5_4-medium/r0.95-s42/study.json
python scripts/summarize_reproduction.py experiments/main/run-plan.json --output experiments/main/summary
```

Run matrix cells **serially** unless you assign additional distinct endpoints:
the templates reuse ports between cells. Mini-DB workers inside one study run
concurrently. Use fresh artifact directories for paper timing. Local optimizer
resume does not reconstruct the duration of a cold run; cross-validation and
full validation are rerun on each invocation. Any `--dry-run` output is synthetic
and must never be treated as an experimental result.

The summary writes `results.json` and `results.csv`, one row per planned cell.
It records missing cells as `missing`, malformed results as `invalid_artifact`,
and leaves absent metrics/times/costs null (blank in CSV). Synthetic cells are
marked explicitly and excluded from measured QPS/recall columns. The full raw
artifact tree remains the source of truth; preserve histories, benchmark output,
model calls, ranking and cross-validation matrices with published results.

The paper's aggregate time sums worker time, not elapsed parallel-stage time.
`paper_aggregate_total_s` is available only for fully instrumented cold studies.
Its `paper_components_s` comprises Mini-DB construction, Mini-DB tuning
(non-LLM local work plus physical cross-view work), CALM inference (all workers),
numerical cross-Mini-DB aggregation and full-database validation. Direct-Full
uses its single-worker invocation time only for complete cold runs, with no
invented five-stage breakdown. Missing usage or unspecified token rates yield
unknown total API cost. No averages, speedups, error bars or baseline values are
fabricated for missing runs.
