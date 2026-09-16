# muTune

muTune tunes vector databases with several independently sampled Mini-DBs and a
constrained LLM optimizer, CALM. It merges complete feasible QPS-recall frontiers,
measures every candidate on every MiniDB, ranks performance and stability,
then measures the top candidates on the original database. The final answer
is the feasible configuration with the highest **measured full-database QPS**.

## Install and inspect

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
# Activate .venv with the command appropriate for your shell.
python -m pip install -e ".[data,dev]"
python scripts/check_static.py
mutune profiles list
mutune validate examples/dry-run.json
mutune render examples/dry-run.json
```

`validate`, `render` and `check_static.py` do not start databases or contact an
LLM. The core only needs Pydantic; the `data` extra adds NumPy, h5py and FAISS.
Database SDKs belong in the external benchmark environment.

For an explicitly synthetic workflow demonstration, use
`mutune tune examples/dry-run.json`. Its metrics are simulated, and must never
be reported as experimental results. `--dry-run` on `tune` also bypasses LLM
calls and service startup. Use a separate artifact directory for demonstrations.

## Prepare Mini-DBs

The HDF5 input must contain dense floating-point `train` and `test` matrices.
Supply your own data; no datasets are redistributed here.

```bash
mutune-build-minidbs --input data/source.hdf5 --output-dir data/minidbs --num-minidbs 3 --sample-ratio 0.1 --bucket-seed 42 --sample-seed 7630 --metric l2 --l2-bucket-width 1.0 --top-k 100
```



## Configure an experiment

Obtain a compatible [vector-db-benchmark checkout](https://github.com/qdrant/vector-db-benchmark)
and install its dependencies in its own environment. The required adapter contract is documented in
[benchmark integration](docs/benchmark-integration.md).

Generate all worker and full-database project files from the data manifest:

```bash
python scripts/prepare_study.py --engine qdrant --manifest data/minidbs/manifest.json --benchmark-repo external/vector-db-benchmark --benchmark-python /absolute/path/to/benchmark-python --output experiments/qdrant
mutune study experiments/qdrant/study.json --validate-only
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
mutune study experiments/qdrant/study.json
python scripts/summarize_study.py experiments/qdrant/artifacts/study
```

The default model label is `gpt-5.4`, with medium reasoning as described in the
paper. Endpoint compatibility and model availability must be checked in the
experimental environment; both are configurable. No API credentials belong in
project JSON. LLM usage is recorded separately for proposals and predictions.
Dollar costs are only calculated when both token rates are explicitly supplied.

Paper Section 5.1 uses QPS and recall as archive/hypervolume coordinates after
applying the recall constraint. These guidance coordinates are separate from
the final QPS objective. Additional efficiency objectives remain available through
explicit `tuning.objectives` for extension experiments; recall is appended to
their guidance space and remains a hard constraint.

`study` runs these stages in order:

1. Independently tune every MiniDB in parallel.
2. Merge and deduplicate the complete feasible local QPS-recall frontiers.
3. Measure every candidate on every MiniDB; reject candidates failing any view.
4. Rank normalized mean performance plus stability (λ defaults to 1).
5. Evaluate the top L candidates on the full database (L defaults to 5).

`mutune tune PROJECT.json` runs only the local optimizer.
`mutune evaluate PROJECT.json --candidate candidate.json --repeat 3` evaluates
fixed configurations. `random` and `knn` are explicit ablations; CALM uses an
LLM for both generation and prediction and does not substitute KNN predictions.

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
