# EIGEN

Source implementation for **EIGEN: Efficient and Generalizable Vector Database
Tuning with Mini-DBs and LLM-Guided Optimization**.

EIGEN tunes vector databases with several independently sampled Mini-DBs and a
constrained LLM optimizer, CALM. It merges complete feasible QPS-recall frontiers,
measures every candidate on every MiniDB, ranks performance and stability,
then measures the top candidates on the original database. The final answer
is the feasible configuration with the highest **measured full-database QPS**.

Start with the [reproduction guide](docs/reproduction.md) for the six datasets,
seven recall targets, supported ablations, sensitivity matrices and result export.

## Install and inspect

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
# Activate .venv with the command appropriate for your shell.
python -m pip install -e ".[data,dev]"
python scripts/check_static.py
python -m unittest discover -s tests
eigen profiles list
eigen validate examples/dry-run.json
eigen render examples/dry-run.json
```

`validate`, `render` and `check_static.py` do not start databases or contact an
LLM. The core only needs Pydantic; the `data` extra adds NumPy, h5py and FAISS.
Database SDKs belong in the external benchmark environment.

For an explicitly synthetic workflow demonstration, use
`eigen tune examples/dry-run.json`. Its metrics are simulated, and must never
be reported as experimental results. `--dry-run` on `tune` also bypasses LLM
calls and service startup. Use a separate artifact directory for demonstrations.

## Prepare Mini-DBs

The HDF5 input must contain dense floating-point `train` and `test` matrices;
the original full database also needs valid exact ground truth for evaluation.
Supply your own data; no datasets are redistributed here. Preserve the dataset
provenance and conversion recipe described in the reproduction guide.

```bash
eigen-build-minidbs --input data/source.hdf5 --output-dir data/minidbs --num-minidbs 3 --sample-ratio 0.1 --bucket-seed 42 --sample-seed 7630 --metric l2 --l2-bucket-width 1.0 --top-k 100
```



## Configure an experiment

Obtain a compatible [vector-db-benchmark checkout](https://github.com/qdrant/vector-db-benchmark)
and install its dependencies in its own environment. The required adapter contract is documented in
[benchmark integration](docs/benchmark-integration.md).

Generate all worker and full-database project files from the data manifest:

```bash
python scripts/prepare_study.py --engine qdrant --manifest data/minidbs/manifest.json --benchmark-repo external/vector-db-benchmark --benchmark-python /absolute/path/to/benchmark-python --output experiments/qdrant
eigen study experiments/qdrant/study.json --validate-only
```

Replace `--benchmark-python` with your benchmark environment's interpreter.
The generator accepts `milvus`, `qdrant`, and `pgvector`; it copies deployment
templates, chooses distinct worker endpoints and pins the benchmark source
content hash. All dataset paths come from the manifest. Configure the correct
dataset description, hardware, concurrency, constraints, search ranges, LLM
endpoint/model and resource allocation before running experiments.

The checked-in `examples/paper/` files illustrate the JSON schema with a
128-dimensional L2 workload. The templates target Milvus 2.3.1, Qdrant 1.16.0 and pgvector 0.6.2. The
PostgreSQL base version and container resource defaults are repository choices.
Each worker has its own database and persistent storage. The provided Compose
configs remove their **own experimental volumes** when a stage stops. They
must be used for dedicated benchmark databases. Parallel workers need adequate
physical CPU, memory and I/O isolation; the resource limits are per database.

## Run in the experimental environment

Set the API key in the environment variable named by `llm.api_key_env` (default
`OPENAI_API_KEY`), then run:

```bash
eigen study experiments/qdrant/study.json
python scripts/summarize_study.py experiments/qdrant/artifacts/study
```

The default model label is `gpt-5.4`, with medium reasoning as described in the
paper. Endpoint compatibility and model availability must be checked in the
experimental environment; both are configurable. LLM usage is recorded separately for proposals and predictions.
Dollar costs are only calculated when both token rates are explicitly supplied.

`study` runs these stages in order:

1. Independently tune every MiniDB in parallel.
2. Merge and deduplicate the complete feasible local QPS-recall frontiers.
3. Measure every candidate on every MiniDB; reject candidates failing any view.
4. Rank normalized mean performance plus stability (λ defaults to 1).
5. Evaluate the top L candidates on the full database (L defaults to 5).

`eigen tune PROJECT.json` runs only the local optimizer.
`eigen evaluate PROJECT.json --candidate candidate.json --repeat 3` evaluates
fixed configurations. `random` and `knn` are explicit ablations.

### Mini-DB workers on separate machines

For the deployment in paper Section 6.1, add `remote_workers` to `study.json`,
with one entry per `minidbs` project in the same order:

```json
"remote_workers": [
  {"host": "eigen-worker-0", "project_config": "/srv/eigen/qdrant-mini-00.json", "python_executable": "/srv/eigen/.venv/bin/python"},
  {"host": "eigen-worker-1", "project_config": "/srv/eigen/qdrant-mini-01.json", "python_executable": "/srv/eigen/.venv/bin/python"},
  {"host": "eigen-worker-2", "project_config": "/srv/eigen/qdrant-mini-02.json", "python_executable": "/srv/eigen/.venv/bin/python"}
]
```

Use distinct SSH hosts/aliases pointing to separate Linux machines with the same
hardware specifications. The coordinator needs `ssh` and SFTP-based `scp`
(OpenSSH 9 or newer), with noninteractive authentication and known host keys.
Configure ports, keys and jump hosts in SSH config. `timeout_s` is optional and
defaults to seven days per SSH/SCP command.

Install this same EIGEN source and the benchmark dependencies on every worker.
Copy each worker's project JSON, deployment files and its exact MiniDB data to
that machine. Update its `artifact_dir`, runner paths (`repo_path`,
`python_executable`, `dataset_path`, `dataset_cache`) and database/Compose
addresses for that machine. Other experiment settings, including the profile,
dataset label, tuning seed, budget, hardware description and LLM configuration,
must match its coordinator project. Set API keys and database credentials in
the environment on each worker; the coordinator does not forward environment
variables.

Run the usual `eigen study study.json`. Both independent tuning and physical
cross-Mini-DB validation execute concurrently over SSH. Loopback database ports
and Compose names may repeat across separate machines. Candidate merging,
ranking and full-database validation execute on the coordinator. The worker
checks the experiment contract and MiniDB checksum before starting evaluations.
`--validate-only` checks coordinator files; it does not contact workers.

Artifacts are copied back into the usual `mini-XX/tuning` and
`mini-XX/validation` directories, so `summarize_study.py` still works. Remote
artifacts remain under the worker project's `artifact_dir/studies/<run-id>/`;
rerunning the same study resumes its local tuning there. Embedded absolute
artifact paths refer to the worker. With `keep_workspace: true`, SCP also copies
benchmark workspaces and their data; set `keep_workspace: false` in both copies
of each project to retain raw logs without those workspace copies. Paper cost
sums measured worker execution times; SSH/SCP overhead appears only in elapsed
pipeline time.
Omit `remote_workers` to use the local thread-based execution.

## Artifacts and verification

The study writes `study_manifest.json`, each worker's `history.jsonl`,
`pareto_archive.json`, `llm/calls.jsonl`, round decisions, the complete
`cross_validation.json` matrix, `ranking.json`, raw benchmark output and
`result.json`. Missing or failed measurements do not become feasible results.
Local tuning resumes only when the workload, profile, optimizer, runner and
LLM contracts match. Cross-validation and full-database measurements are rerun
on each invocation; resumed wall-clock time is not the duration of a fresh run.

Each local `result.json` has the full feasible QPS-recall `pareto_candidates`
frontier. `transfer_candidates` is a compatibility output containing that same
frontier, with no per-region cap. The old `transfer_candidates_per_region`
setting is accepted but deprecated and ignored. Region and batch hypervolume
use those same guidance coordinates; final selection still maximizes feasible QPS.

CALM initialization defaults to one physical evaluation per conditional region
(`initial_samples: null`), hence seven on the bundled Milvus profile. Local
budget 20 includes initialization and failed database evaluations. Cross-Mini-DB
measurements and final shortlist validation are counted separately.

`paper_aggregate_total_s` sums worker costs, following Sections 6.1 and 6.7.
Its five components separate construction, physical Mini-DB work, LLM inference,
numerical aggregation and full-database validation. Elapsed pipeline time remains
a separate diagnostic.
