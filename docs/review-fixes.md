# 2026-09-15 code review fixes

This change addresses the six findings in the paper/code review. It changes
measurement and transfer behavior, so existing experiment numbers need reruns.

| Finding | Corrected behavior | Offline evidence |
| --- | --- | --- |
| Milvus server knobs were mounted in an unread file | Complete versioned defaults plus candidate overrides are mounted at `/milvus/configs/milvus.yaml`; mount verification is explicitly distinguished from runtime readback | Lifecycle tests check defaults, overridden values, correct mount, mismatch rejection and verification metadata |
| Duplicate Milvus port keyword | Private configure/upload/search snapshots pass one normalized port and preserve other connection parameters | All three connection paths tested with explicit/default ports; input dictionaries are unchanged |
| Filtered Recall denominator and empty answers | Denominator is the available exact top-K answer count; an empty answer scores 1 only for an empty returned set | Short, full, empty, wrong-empty, duplicate and missing-GT cases |
| Skipped builds could be accepted as fresh | Manual skip flags are rejected; fresh evaluation requires a matching upload result with finite, nonnegative `total_time` | Runner tests exercise process result ingestion with missing, incomplete, invalid and valid build evidence |
| Single-objective local frontier discarded useful transfer backups | Keep the true optimization frontier and add feasible per-region speed/recall representatives to a distinct transfer pool | A/C are locally fastest but fail on the other view; B has slightly lower QPS and passes both views, and now survives |
| Construction omitted and parallel costs ambiguous | Record construction stage costs, current-invocation LLM duration, resumed evaluation counts, parallel critical path and conditional cold-start totals | Disjoint construction stages, overlapping-worker accounting, resumed histories and absent timing metadata |

The transfer pool is an explicit refinement of paper Section 5.3's frontier-only
description. Default optimization still maximizes QPS subject to recall;
recall has not become an optimization objective and build time remains a
diagnostic. Update the paper's candidate-collection description to mention the
frontier plus up to three feasible representatives per structural region
(highest QPS, highest recall, then QPS order). Deduplication, all-view feasibility,
normalization, stability scoring and top-L full-database selection are unchanged.

Construction timing begins inside the builder and ends after data output,
ground truth and checksum preparation, immediately before writing its manifest.
Both HDF5 and geo builders record it. The geo ground-truth stage includes query
output; the final manifest-preparation stage includes the source checksum.

`total_invocation_wall_s` covers only the current study call, including service
startup/teardown and fixed evaluations. `cold_start_pipeline_wall_s` adds the
recorded construction duration to that invocation only when every worker starts
with zero historical evaluations. It is an accounted sequential pipeline cost,
not elapsed time between two separately launched commands. It excludes idle
gaps and final timing-manifest/result writes. Old manifests yield `null` rather
than an invented construction duration; resumed runs yield no cold-start total.

The five `additive_wall_s` components sum to that pipeline cost. Within parallel
tuning, LLM time is the measured LLM time of the last-finishing worker; the rest
of the parallel stage is assigned to MiniDB tuning. This is a stated critical
path attribution, not the sum of simultaneously running workers' LLM calls.
The cross-validation/aggregation/control component includes coordinator work
outside tuning and full validation. `llm_worker_sum_s` remains a separate
diagnostic. This convention should accompany any paper timing breakdown.

Use new artifact directories for corrected experiments. Regenerate MiniDB
manifests to measure construction; do not attach guessed durations to old data.
The runner version and tuning contract changed, so old histories are not
silently resumed into the corrected measurement contract. Milvus example and
generated project configurations now use `milvus_yaml`; existing names remain
compatible aliases. No original data or uploaded paper is modified.

Validation commands (no database or LLM calls):

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/check_static.py
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

The repair environment has no Docker executable. Real Milvus startup, native
index support, internal knob readback, full-data experiments and published
performance claims remain unverified. Before using new experiment numbers,
run a small fresh-build/search cycle in the pinned database environment, check
the produced configuration/result artifacts, then rerun the study.

Repair verification: all 124 offline unit tests passed; static validation passed
for 7 profiles, 21 regions, 13 project schemas and 3 study schemas; Ruff lint and
format checks passed. The compatibility edits were additionally exercised on a
private copy of actual benchmark commit
`e8299454a07d9c429cd1cce9ab610fea68205e44`: all three Milvus connection expressions
accepted explicit/default ports, and the reviewed Recall cases produced
`[1.0, 0.0, 1.0]`. This check used stub database calls. A built 0.3.0 wheel also
loaded the packaged resources and rendered the complete Milvus configuration
successfully from outside the checkout.
