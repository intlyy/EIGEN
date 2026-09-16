"""Narrow, checked compatibility edits applied only to private benchmark copies."""

from __future__ import annotations

import ast
from importlib import resources
from pathlib import Path

from mutune.errors import RunnerError

RECALL_CONTRACT = "available-ground-truth-v1; empty-correct-only-if-returned-empty"
MILVUS_GEO_CONTRACT = "payload-id-prefilter-v1"


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


def install_milvus_geo_compatibility(workspace: Path) -> list[str]:
    """Wrap reviewed Milvus interfaces without replacing its uploader/index code."""
    directory = workspace / "engine/clients/milvus"
    sources = {}
    contracts = {
        "configure": ("MilvusConfigurator", "recreate", ["self", "dataset", "collection_params"]),
        "search": ("MilvusSearcher", "search_one", ["cls", "query", "top"]),
    }
    for name, (class_name, method_name, arguments) in contracts.items():
        path = directory / f"{name}.py"
        tree = _module(path)
        methods = [
            method
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
            for method in node.body
            if isinstance(method, ast.FunctionDef) and method.name == method_name
        ]
        if len(methods) != 1 or [a.arg for a in methods[0].args.args] != arguments:
            raise RunnerError(
                f"unsupported Milvus geo adapter interface: {class_name}.{method_name}"
            )
        sources[name] = path.read_bytes()
    helpers = [directory / f"mutune_{name}_base.py" for name in contracts]
    helpers.append(directory / "mutune_geo.py")
    if any(path.exists() or path.is_symlink() for path in helpers):
        raise RunnerError("Milvus geo helper paths already exist in benchmark source")
    package = resources.files("mutune.resources.vectordb_benchmark")
    changed = []
    for name, source in sources.items():
        base = directory / f"mutune_{name}_base.py"
        base.write_bytes(source)
        wrapper = directory / f"{name}.py"
        wrapper.write_bytes(package.joinpath(f"milvus_geo_{name}.py.txt").read_bytes())
        changed.extend(
            [base.relative_to(workspace).as_posix(), wrapper.relative_to(workspace).as_posix()]
        )
    helper = directory / "mutune_geo.py"
    helper.write_bytes(resources.files("mutune").joinpath("geo.py").read_bytes())
    changed.append(helper.relative_to(workspace).as_posix())
    return changed
