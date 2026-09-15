"""Narrow, checked compatibility edits applied only to private benchmark copies."""

from __future__ import annotations

import ast
from importlib import resources
from pathlib import Path

from mutune.errors import RunnerError

RECALL_CONTRACT = "available-ground-truth-v1; empty-correct-only-if-returned-empty"


def _module(path: Path) -> ast.Module:
    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"benchmark compatibility requires a regular file: {path}")
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeError) as error:
        raise RunnerError(f"cannot parse benchmark compatibility target: {path}") from error


def install_benchmark_compatibility(workspace: Path, engine: str) -> list[str]:
    """Preserve upstream classes and decorators; replace only reviewed expressions."""
    relative = Path("engine/base_client/search.py")
    path = workspace / relative
    tree = _module(path)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BaseSearcher"]
    methods = [
        n
        for c in classes
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "_search_one"
    ]
    if len(methods) != 1 or [a.arg for a in methods[0].args.args] != ["cls", "query", "top"]:
        raise RunnerError("unsupported BaseSearcher._search_one signature")
    replacement = (
        resources.files("mutune.resources.vectordb_benchmark")
        .joinpath("benchmark_search_one.py.txt")
        .read_text(encoding="utf-8")
    )
    methods[0].body = ast.parse(replacement).body[0].body
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")
    changed = [relative.as_posix()]
    if engine == "milvus":
        for name in ("configure", "upload", "search"):
            relative = Path(f"engine/clients/milvus/{name}.py")
            path = workspace / relative
            tree = _module(path)
            calls = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "connect"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "connections"
            ]
            if not calls:
                raise RunnerError(f"unsupported Milvus connection adapter: {relative}")
            patched = False
            for call in calls:
                ports = [k for k in call.keywords if k.arg == "port"]
                unpack = [
                    k
                    for k in call.keywords
                    if k.arg is None
                    and isinstance(k.value, ast.Name)
                    and k.value.id == "connection_params"
                ]
                if ports and unpack:
                    # The explicit, normalized port replaces the dictionary's
                    # port exactly once. Do not mutate shared connection_params.
                    unpack[0].value = ast.Dict(
                        keys=[None, ast.Constant("port")], values=[unpack[0].value, ports[0].value]
                    )
                    call.keywords.remove(ports[0])
                    patched = True
            if patched:
                path.write_text(
                    ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8"
                )
                changed.append(relative.as_posix())
    return changed
