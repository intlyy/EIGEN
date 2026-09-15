from __future__ import annotations

import sys
import tempfile
import types
import unittest
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mutune.errors import RunnerError
from mutune.runners.vectordb_benchmark import _install_engine_overlays

_OVERLAY_PACKAGE = "mutune.resources.vectordb_benchmark"


class _IncompatibilityError(Exception):
    pass


class _Distance:
    L2 = "l2"
    COSINE = "cosine"
    DOT = "dot"


class _FakeCursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, object | None, dict]] = []
        self.closed = False

    def execute(self, statement, params=None, **kwargs) -> None:
        self.statements.append((statement, params, kwargs))

    def fetchall(self):
        return [(1, 0.0)]

    def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, object | None]] = []
        self.cursor_instance = _FakeCursor()
        self.closed = False

    def execute(self, statement, params=None) -> None:
        self.statements.append((statement, params))

    def cursor(self) -> _FakeCursor:
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


def _module(name: str, **attributes) -> types.ModuleType:
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    return result


def _stub_modules(connection: _FakeConnection) -> dict[str, types.ModuleType]:
    class _BaseConfigurator:
        def __init__(self, host, collection_params, connection_params) -> None:
            self.host = host
            self.collection_params = collection_params
            self.connection_params = connection_params

    class _BaseUploader:
        pass

    class _BaseSearcher:
        pass

    class _ConditionParser:
        pass

    numpy = _module("numpy", array=lambda value: value)
    psycopg = _module("psycopg", connect=lambda **kwargs: connection)
    pgvector = _module("pgvector")
    pgvector_psycopg = _module("pgvector.psycopg", register_vector=lambda target: None)
    pgvector.psycopg = pgvector_psycopg
    dataset_reader = _module("dataset_reader")
    dataset_reader_base = _module("dataset_reader.base_reader", Record=object, Query=object)
    engine = _module("engine")
    base_client = _module("engine.base_client", IncompatibilityError=_IncompatibilityError)
    benchmark = _module("benchmark")
    benchmark_dataset = _module("benchmark.dataset", Dataset=object)
    configure = _module("engine.base_client.configure", BaseConfigurator=_BaseConfigurator)
    distances = _module("engine.base_client.distances", Distance=_Distance)
    upload = _module("engine.base_client.upload", BaseUploader=_BaseUploader)
    search = _module("engine.base_client.search", BaseSearcher=_BaseSearcher)
    clients = _module("engine.clients")
    pgvector_client = _module("engine.clients.pgvector")
    config = _module(
        "engine.clients.pgvector.config",
        get_db_config=lambda host, params: {"host": host, **params},
    )
    parser = _module("engine.clients.pgvector.parser", PgVectorConditionParser=_ConditionParser)
    return {
        item.__name__: item
        for item in (
            numpy,
            psycopg,
            pgvector,
            pgvector_psycopg,
            dataset_reader,
            dataset_reader_base,
            benchmark,
            benchmark_dataset,
            engine,
            base_client,
            configure,
            distances,
            upload,
            search,
            clients,
            pgvector_client,
            config,
            parser,
        )
    }


def _load_overlay(resource_name: str, connection: _FakeConnection):
    source = resources.files(_OVERLAY_PACKAGE).joinpath(resource_name).read_text(encoding="utf-8")
    module = types.ModuleType(f"test_overlay_{resource_name}")
    with patch.dict(sys.modules, _stub_modules(connection)):
        exec(compile(source, resource_name, "exec"), module.__dict__)
    return module


class PgvectorOverlayTests(unittest.TestCase):
    def test_overlay_is_installed_only_in_isolated_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            workspace = root / "workspace"
            relative_root = Path("engine/clients/pgvector")
            (source / relative_root).mkdir(parents=True)
            (workspace / relative_root).mkdir(parents=True)
            for name in ("configure.py", "upload.py", "search.py"):
                (source / relative_root / name).write_text("source-sentinel")
                (workspace / relative_root / name).write_text("workspace-sentinel")

            installed = _install_engine_overlays(workspace, "pgvector")

            self.assertEqual(
                installed,
                [
                    "engine/clients/pgvector/configure.py",
                    "engine/clients/pgvector/upload.py",
                    "engine/clients/pgvector/search.py",
                ],
            )
            self.assertEqual((source / relative_root / "upload.py").read_text(), "source-sentinel")
            self.assertIn(
                "class PgVectorUploader",
                (workspace / relative_root / "upload.py").read_text(),
            )
            self.assertIn(
                "class PgVectorSearcher",
                (workspace / relative_root / "search.py").read_text(),
            )
            self.assertIn(
                "CREATE TABLE",
                (workspace / relative_root / "configure.py").read_text(),
            )
            self.assertEqual(_install_engine_overlays(workspace, "qdrant"), [])

    def test_configurator_creates_a_logged_table_for_wal_knobs(self) -> None:
        connection = _FakeConnection()
        overlay = _load_overlay("pgvector_configure.py.txt", connection)
        configurator = overlay.PgVectorConfigurator("localhost", {}, {})
        dataset = SimpleNamespace(config=SimpleNamespace(distance=_Distance.L2, vector_size=384))

        configurator.recreate(dataset, {})

        self.assertTrue(
            any("CREATE TABLE items" in statement for statement, _params in connection.statements)
        )

    def test_overlay_fails_closed_for_incompatible_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RunnerError, "expected a regular file"):
                _install_engine_overlays(Path(temporary), "pgvector")

    def test_uploader_supports_every_pgvector_index_mode(self) -> None:
        connection = _FakeConnection()
        overlay = _load_overlay("pgvector_upload.py.txt", connection)
        uploader = overlay.PgVectorUploader

        uploader.conn = connection
        uploader.upload_params = {
            "index_type": "hnsw",
            "hnsw_config": {"m": 24, "ef_construct": 160},
        }
        result = uploader.post_upload(_Distance.L2)
        self.assertEqual(result, {"index_type": "hnsw", "index_created": True})
        self.assertTrue(any("USING hnsw" in statement for statement, _ in connection.statements))

        connection.statements.clear()
        uploader.upload_params = {
            "index_type": "ivf_flat",
            "ivfflat_config": {"lists": 512},
        }
        result = uploader.post_upload(_Distance.COSINE)
        self.assertEqual(result, {"index_type": "ivfflat", "index_created": True})
        self.assertTrue(
            any(
                "USING ivfflat" in statement and "lists = 512" in statement
                for statement, _ in connection.statements
            )
        )

        connection.statements.clear()
        uploader.upload_params = {"index_type": "exact"}
        result = uploader.post_upload(_Distance.L2)
        self.assertEqual(result, {"index_type": "exact", "index_created": False})
        self.assertFalse(any("CREATE INDEX" in statement for statement, _ in connection.statements))

    def test_uploader_rebuild_reuses_rows_and_replaces_only_vector_index(self) -> None:
        connection = _FakeConnection()
        overlay = _load_overlay("pgvector_upload.py.txt", connection)
        uploader = overlay.PgVectorUploader
        uploader.conn = connection
        uploader.upload_params = {
            "index_type": "hnsw",
            "hnsw_config": {"m": 16, "ef_construct": 128},
        }

        result = uploader.rebuild_index(_Distance.L2)

        self.assertEqual(result, {"index_type": "hnsw", "index_created": True})
        statements = [statement for statement, _params in connection.statements]
        self.assertEqual(statements[0], "DROP INDEX IF EXISTS items_embedding_idx")
        self.assertIn("CREATE INDEX ON items USING hnsw", statements[1])

    def test_searcher_applies_index_specific_runtime_setting(self) -> None:
        connection = _FakeConnection()
        overlay = _load_overlay("pgvector_search.py.txt", connection)
        searcher = overlay.PgVectorSearcher

        searcher.init_client(
            "localhost",
            _Distance.L2,
            {},
            {"config": {"index_type": "hnsw", "hnsw_ef": 200}},
        )
        self.assertIn(
            ("SELECT set_config('hnsw.ef_search', %s, false)", ("200",), {}),
            connection.cursor_instance.statements,
        )

        connection = _FakeConnection()
        overlay.psycopg.connect = lambda **kwargs: connection
        searcher.init_client(
            "localhost",
            _Distance.COSINE,
            {},
            {"config": {"index_type": "ivfflat", "ivfflat_probes": 64}},
        )
        self.assertIn(
            ("SELECT set_config('ivfflat.probes', %s, false)", ("64",), {}),
            connection.cursor_instance.statements,
        )

        connection = _FakeConnection()
        overlay.psycopg.connect = lambda **kwargs: connection
        searcher.init_client(
            "localhost",
            _Distance.L2,
            {},
            {"config": {"index_type": "exact", "exact_parallel_workers": 4}},
        )
        statements = [item[0] for item in connection.cursor_instance.statements]
        self.assertIn(
            (
                "SELECT set_config('max_parallel_workers_per_gather', %s, false)",
                ("4",),
                {},
            ),
            connection.cursor_instance.statements,
        )
        self.assertIn("SET enable_indexscan = off", statements)
        self.assertNotIn("SELECT set_config('hnsw.ef_search', %s, false)", statements)
        self.assertNotIn("SELECT set_config('ivfflat.probes', %s, false)", statements)

    def test_overlay_rejects_unknown_index_type_and_non_integer_parameters(self) -> None:
        connection = _FakeConnection()
        overlay = _load_overlay("pgvector_upload.py.txt", connection)
        uploader = overlay.PgVectorUploader
        uploader.conn = connection

        uploader.upload_params = {"index_type": "diskann"}
        with self.assertRaisesRegex(_IncompatibilityError, "Unsupported"):
            uploader.post_upload(_Distance.L2)

        uploader.upload_params = {
            "index_type": "hnsw",
            "hnsw_config": {"m": "16", "ef_construct": 128},
        }
        with self.assertRaisesRegex(_IncompatibilityError, "positive integer"):
            uploader.post_upload(_Distance.L2)


if __name__ == "__main__":
    unittest.main()
