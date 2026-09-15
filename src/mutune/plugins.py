"""Lazy discovery and construction of evaluation runner plugins.

Third-party runners are registered in the ``mutune.runners`` entry-point
group.  Discovery deliberately does not import plugin modules: a selected
entry point is loaded only when :func:`load_runner_class` or
:func:`create_runner` is called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module, metadata
from typing import Any, Callable, Iterable

from mutune.api import BaseRunner, RunnerContext
from mutune.errors import PluginError

RUNNER_ENTRY_POINT_GROUP = "mutune.runners"

# String targets keep built-ins lazy too.  In particular, importing the
# vector-db-benchmark runner must not import or require any database clients.
BUILTIN_RUNNERS: dict[str, str] = {
    "dry-run": "mutune.runners.dry_run:DryRunRunner",
    "vector-db-benchmark": ("mutune.runners.vectordb_benchmark:VectorDBBenchmarkRunner"),
}


@dataclass(frozen=True, slots=True)
class RunnerPlugin:
    """A discovered runner without an eagerly imported implementation."""

    plugin_id: str
    source: str
    distribution: str | None = None
    distribution_version: str | None = None
    _loader: Callable[[], Any] = field(repr=False, compare=False, default=lambda: None)

    def load_class(self) -> type[BaseRunner]:
        """Load and validate the runner class represented by this record."""

        try:
            candidate = self._loader()
        except Exception as error:  # pragma: no cover - exact import errors vary
            raise PluginError(
                f"Failed to load runner plugin {self.plugin_id!r} from {self.source}: {error}"
            ) from error

        if not isinstance(candidate, type) or not issubclass(candidate, BaseRunner):
            raise PluginError(
                f"Runner plugin {self.plugin_id!r} must resolve to a BaseRunner subclass"
            )

        api_version = str(getattr(candidate, "API_VERSION", ""))
        if api_version != BaseRunner.API_VERSION:
            raise PluginError(
                f"Runner plugin {self.plugin_id!r} uses API {api_version!r}; "
                f"muTune requires {BaseRunner.API_VERSION!r}"
            )

        declared_id = str(getattr(candidate, "PLUGIN_ID", ""))
        if declared_id != self.plugin_id:
            raise PluginError(
                f"Runner entry point {self.plugin_id!r} declares PLUGIN_ID {declared_id!r}"
            )
        return candidate


def _load_string_target(target: str) -> Any:
    module_name, separator, attribute_name = target.partition(":")
    if not separator or not module_name or not attribute_name:
        raise PluginError(f"Invalid built-in runner target: {target!r}")
    module = import_module(module_name)
    candidate: Any = module
    for component in attribute_name.split("."):
        candidate = getattr(candidate, component)
    return candidate


def _entry_points() -> Iterable[metadata.EntryPoint]:
    """Return runner entry points across supported importlib APIs."""

    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return discovered.select(group=RUNNER_ENTRY_POINT_GROUP)
    # Python 3.9 compatibility for downstream users, despite muTune itself
    # requiring 3.11+.
    return discovered.get(RUNNER_ENTRY_POINT_GROUP, ())  # type: ignore[union-attr]


def _entry_point_distribution(
    entry_point: metadata.EntryPoint,
) -> tuple[str | None, str | None]:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return None, None
    name = getattr(distribution, "name", None)
    version = getattr(distribution, "version", None)
    return (
        str(name) if name is not None else None,
        str(version) if version is not None else None,
    )


def discover_runners(*, include_builtins: bool = True) -> dict[str, RunnerPlugin]:
    """Discover available runner plugins without importing their modules.

    Duplicate IDs are rejected rather than resolved by installation order.
    Installed packages also cannot shadow built-in runner IDs.
    """

    plugins: dict[str, RunnerPlugin] = {}
    if include_builtins:
        for plugin_id, target in BUILTIN_RUNNERS.items():
            plugins[plugin_id] = RunnerPlugin(
                plugin_id=plugin_id,
                source=f"builtin:{target}",
                _loader=lambda target=target: _load_string_target(target),
            )

    for entry_point in _entry_points():
        plugin_id = entry_point.name
        if plugin_id in plugins:
            raise PluginError(
                f"Duplicate runner plugin ID {plugin_id!r}: "
                f"{plugins[plugin_id].source} and {entry_point.value}"
            )
        distribution, version = _entry_point_distribution(entry_point)
        plugins[plugin_id] = RunnerPlugin(
            plugin_id=plugin_id,
            source=f"entry-point:{entry_point.value}",
            distribution=distribution,
            distribution_version=version,
            _loader=entry_point.load,
        )

    return dict(sorted(plugins.items()))


def load_runner_class(plugin_id: str) -> type[BaseRunner]:
    """Load one runner class by ID."""

    plugins = discover_runners()
    try:
        plugin = plugins[plugin_id]
    except KeyError as error:
        available = ", ".join(plugins) or "none"
        raise PluginError(
            f"Unknown runner plugin {plugin_id!r}; available runners: {available}"
        ) from error
    return plugin.load_class()


def create_runner(plugin_id: str, context: RunnerContext) -> BaseRunner:
    """Construct a selected runner using the stable API-v1 context."""

    runner_class = load_runner_class(plugin_id)
    try:
        runner = runner_class(context)
    except Exception as error:
        raise PluginError(f"Failed to construct runner plugin {plugin_id!r}: {error}") from error
    return runner


__all__ = [
    "BUILTIN_RUNNERS",
    "RUNNER_ENTRY_POINT_GROUP",
    "RunnerPlugin",
    "create_runner",
    "discover_runners",
    "load_runner_class",
]
