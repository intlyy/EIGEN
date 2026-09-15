"""Small dependency-free helpers shared across muTune components."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def dataset_sha256(path: str | Path) -> str:
    """Content identity of HDF5 or the three benchmark input files of a directory."""
    path = Path(path)
    files = (
        [
            path / name
            for name in ("vectors.npy", "payloads.jsonl", "tests.jsonl")
            if (path / name).is_file()
        ]
        if path.is_dir()
        else [path]
    )
    digest = hashlib.sha256()
    for file in files:
        if path.is_dir():
            digest.update(file.name.encode() + b"\0")
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_.-]+")


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: Any, length: int = 16) -> str:
    """Return a stable SHA-256 prefix for JSON-compatible data."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def safe_name(value: str, max_length: int = 100) -> str:
    """Convert an arbitrary label to a portable file/experiment name."""

    normalized = _SAFE_NAME.sub("-", value.strip()).strip("-._")
    if not normalized:
        normalized = "run"
    return normalized[:max_length]


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file in its target directory."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def ensure_within(path: Path, root: Path) -> Path:
    """Resolve ``path`` and reject traversal or symlink escape outside ``root``."""

    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Path escapes allowed root: {resolved_path} (root: {resolved_root})")
    return resolved_path
