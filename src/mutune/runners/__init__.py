"""Built-in muTune runners, exposed without eager sibling imports."""

from __future__ import annotations

from typing import Any

__all__ = ["DryRunRunner", "VectorDBBenchmarkRunner"]


def __getattr__(name: str) -> Any:
    if name == "DryRunRunner":
        from mutune.runners.dry_run import DryRunRunner

        return DryRunRunner
    if name == "VectorDBBenchmarkRunner":
        from mutune.runners.vectordb_benchmark import VectorDBBenchmarkRunner

        return VectorDBBenchmarkRunner
    raise AttributeError(name)
