"""Create a reviewable source archive; excludes data, secrets, caches and run artifacts."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/mutune-paper-source.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    included = [
        root / name
        for name in (
            "README.md",
            "CHANGELOG.md",
            "LICENSE-PENDING.md",
            "pyproject.toml",
            ".gitignore",
            "build_minidb.py",
            "build_geo_radius_minidb.py",
            "build_tiny1m.py",
        )
    ]
    for directory in ("src/mutune", "tests", "scripts", "docs", "examples", ".github"):
        included.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and not any(
                part in {"__pycache__", "artifacts", ".venv", ".ruff_cache"}
                for part in path.relative_to(root).parts
            )
            and path.suffix not in {".pyc", ".pyo"}
        )
    if output in included:
        raise ValueError("archive output cannot replace a source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(included):
            archive.write(path, "mutune/" + path.relative_to(root).as_posix())
    print(f"{output} ({len(included)} source files)")


if __name__ == "__main__":
    main()
