"""Package an explicit source-file selection, excluding environments and run artifacts."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT_FILES = (
    "README.md",
    "CHANGELOG.md",
    "LICENSE-PENDING.md",
    "CITATION.cff",
    "pyproject.toml",
    ".gitignore",
    "build_minidb.py",
    "build_geo_radius_minidb.py",
    "build_tiny1m.py",
)
EXCLUDED = {
    "__pycache__",
    "artifacts",
    ".venv",
    ".ruff_cache",
    ".pytest_cache",
    ".git",
    "data",
    "output",
    "experiments",
    "reproduction",
    "node_modules",
}
SOURCE_SUFFIXES = {".py", ".json", ".md", ".toml", ".yml", ".yaml", ".txt"}


def source_files(root: Path) -> list[Path]:
    root = root.resolve()
    included = [root / name for name in ROOT_FILES]
    missing = [path.name for path in included if not path.is_file()]
    if missing:
        raise ValueError(f"required publication files are missing: {', '.join(missing)}")
    for directory in ("src/eigen", "tests", "scripts", "docs", "examples", ".github"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            if any(part in EXCLUDED or part.startswith(".env") for part in relative.parts):
                continue
            if (
                path.suffix in SOURCE_SUFFIXES
                or path.name == "Dockerfile"
                or path.suffix == ".Dockerfile"
                or path.name == "MILVUS-LICENSE"
                or relative.as_posix() == "docs/EIGEN_VLDB.pdf"
            ):
                included.append(path)
    return sorted(set(included))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/EIGEN-paper-source.zip"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    included = source_files(root)
    if output in included:
        raise ValueError("archive output cannot replace a source file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w" if args.overwrite else "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(included):
            archive.write(path, "EIGEN/" + path.relative_to(root).as_posix())
    print(f"{output} ({len(included)} source files)")


if __name__ == "__main__":
    main()
