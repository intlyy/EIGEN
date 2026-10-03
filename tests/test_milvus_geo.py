from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from eigen.benchmark_compat import install_milvus_geo_compatibility
from eigen.errors import RunnerError
from eigen.geo import (
    DEFAULT_MAX_FILTER_BYTES,
    EARTH_RADIUS_METERS,
    GeoRadiusFilter,
    _build_geo_columns,
    _condition_mask,
    _geo_mask,
)

_OVERLAY_PACKAGE = "eigen.resources.vectordb_benchmark"


def _clause(lat=0.0, lon=0.0, radius=150_000.0, field="location"):
    return {field: {"geo": {"lat": lat, "lon": lon, "radius": radius}}}


def _payload(lat, lon):
    return {"location": {"lat": lat, "lon": lon}}


def _dataset(path, payloads, vector_count=None):
    count = len(payloads) if vector_count is None else vector_count
    np.save(path / "vectors.npy", np.arange(count * 2, dtype=np.float32).reshape(count, 2))
    (path / "payloads.jsonl").write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads), encoding="utf-8"
    )
    return path


def _expression_ids(expression, count):
    if expression == "id < 0":
        return []
    if expression == "id >= 0":
        return list(range(count))
    if expression.startswith("id in "):
        return json.loads(expression[len("id in ") :])
    raise AssertionError(f"Unexpected primary-key expression: {expression}")


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _load_overlay(resource_name, base_class):
    modules = {
        name: _module(name) for name in ("engine", "engine.clients", "engine.clients.milvus")
    }
    modules.update(
        {
            "engine.clients.milvus.eigen_configure_base": _module(
                "engine.clients.milvus.eigen_configure_base", MilvusConfigurator=base_class
            ),
            "engine.clients.milvus.eigen_search_base": _module(
                "engine.clients.milvus.eigen_search_base", MilvusSearcher=base_class
            ),
            "engine.clients.milvus.eigen_geo": _module(
                "engine.clients.milvus.eigen_geo", GeoRadiusFilter=GeoRadiusFilter
            ),
        }
    )
    module = types.ModuleType(f"test_{resource_name}")
    source = resources.files(_OVERLAY_PACKAGE).joinpath(resource_name).read_text(encoding="utf-8")
    with patch.dict(sys.modules, modules):
        exec(compile(source, resource_name, "exec"), module.__dict__)
    return module


def _searcher(collection):
    class BaseSearcher:
        initialized = []

        @classmethod
        def init_client(cls, host, distance, connection_params, search_params):
            cls.initialized.append((host, distance, connection_params, search_params))
            cls.collection = collection
            cls.distance = {"l2": "L2", "cosine": "IP"}[distance]
            cls.search_params = search_params

    return _load_overlay("milvus_geo_search.py.txt", BaseSearcher).MilvusSearcher


class GeoRadiusFilterTests(unittest.TestCase):
    def test_payload_row_ids_and_shared_ground_truth_predicate(self):
        payloads = [_payload(0, 4), _payload(0, 0), _payload(0, 1), _payload(0, -3)]
        conditions = {"and": [_clause()]}
        with tempfile.TemporaryDirectory() as temporary:
            dataset = _dataset(Path(temporary), payloads)
            predicate = GeoRadiusFilter.from_dataset(dataset, {"location": "geo"})
            self.assertEqual(predicate.max_filter_bytes, DEFAULT_MAX_FILTER_BYTES)
            ids = _expression_ids(predicate.expression(conditions), len(payloads))

        self.assertEqual(ids, [1, 2])
        expected = _condition_mask(conditions, _build_geo_columns(payloads), len(payloads))
        np.testing.assert_array_equal(ids, np.flatnonzero(expected))

    def test_dateline_and_pole_are_spherical_not_flat_coordinate_ranges(self):
        payloads = [_payload(0, -179.9), _payload(0, 179.8), _payload(0, 0)]
        columns = _build_geo_columns(payloads)
        mask = _condition_mask({"and": [_clause(lon=179.9, radius=30_000)]}, columns, 3)
        np.testing.assert_array_equal(mask, [True, True, False])

        columns = _build_geo_columns([_payload(90, -120), _payload(90, 80), _payload(89, 0)])
        mask = _condition_mask({"and": [_clause(lat=90, radius=1)]}, columns, 3)
        np.testing.assert_array_equal(mask, [True, True, False])

    def test_radius_boundary_is_strict_and_zero_radius_has_no_matches(self):
        latitudes = np.radians(np.array([0.0, 0.0, 0.0]))
        longitudes = np.radians(np.array([0.0, 1.0, 2.0]))
        radius = EARTH_RADIUS_METERS * np.radians(1.0)
        mask = _geo_mask(latitudes, longitudes, {"lat": 0, "lon": 0, "radius": radius})
        np.testing.assert_array_equal(mask, [True, False, False])
        mask = _geo_mask(latitudes, longitudes, {"lat": 0, "lon": 0, "radius": 0})
        self.assertFalse(mask.any())

    def test_and_or_groups_are_both_applied_and_fields_are_separate_clauses(self):
        payloads = [_payload(0, x) for x in (0, 1, 2, 5)]
        for payload in payloads:
            payload["other"] = dict(payload["location"])
        columns = _build_geo_columns(payloads)
        conditions = {
            "and": [_clause(lon=0, radius=170_000)],
            "or": [_clause(lon=1, radius=1), _clause(lon=5, radius=1)],
        }
        np.testing.assert_array_equal(_condition_mask(conditions, columns, 4), [0, 1, 0, 0])
        conditions = {"or": [{**_clause(radius=1), **_clause(lon=2, radius=1, field="other")}]}
        np.testing.assert_array_equal(_condition_mask(conditions, columns, 4), [1, 0, 1, 0])

    def test_conditions_are_recomputed_for_each_query_including_all_and_empty(self):
        columns = _build_geo_columns([_payload(0, 0), _payload(0, 1), _payload(0, 2)])
        predicate = GeoRadiusFilter(columns, 3)
        conditions = {"and": [_clause(radius=1)]}
        self.assertEqual(_expression_ids(predicate.expression(conditions), 3), [0])
        conditions["and"][0]["location"]["geo"]["lon"] = 2
        self.assertEqual(_expression_ids(predicate.expression(conditions), 3), [2])
        conditions["and"][0]["location"]["geo"]["radius"] = 0
        self.assertEqual(_expression_ids(predicate.expression(conditions), 3), [])
        self.assertEqual(_expression_ids(predicate.expression(None), 3), [0, 1, 2])
        self.assertEqual(_expression_ids(predicate.expression({}), 3), [0, 1, 2])

    def test_invalid_schema_and_missing_declared_payload_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = _dataset(Path(temporary), [_payload(0, 0)])
            for schema in ({}, {"location": "float"}, {"location": "geo", "tag": "int"}, []):
                with self.subTest(schema=schema), self.assertRaises(ValueError):
                    GeoRadiusFilter.from_dataset(dataset, schema)
            with self.assertRaisesRegex(ValueError, "missing"):
                GeoRadiusFilter.from_dataset(dataset, {"missing": "geo"})
            (dataset / "payloads.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                GeoRadiusFilter.from_dataset(dataset, {"location": "geo"})

    def test_payload_count_shape_and_coordinates_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            for payloads, count in (([_payload(0, 0)], 2), ([_payload(0, 0)] * 2, 1)):
                _dataset(dataset, payloads, count)
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, "rows"):
                    GeoRadiusFilter.from_dataset(dataset, {"location": "geo"})
            for payload in (_payload(91, 0), _payload(0, 181), _payload(float("nan"), 0)):
                _dataset(dataset, [payload])
                with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "Invalid"):
                    GeoRadiusFilter.from_dataset(dataset, {"location": "geo"})
            _dataset(dataset, [_payload(0, 0), {"location": {"lat": 0}}])
            with self.assertRaisesRegex(ValueError, "Invalid geo payload"):
                GeoRadiusFilter.from_dataset(dataset, {"location": "geo"})
            np.save(dataset / "vectors.npy", np.zeros(2))
            with self.assertRaisesRegex(ValueError, "matrix"):
                GeoRadiusFilter.from_dataset(dataset, {"location": "geo"})

    def test_unsupported_predicates_and_invalid_centers_fail_closed(self):
        columns = _build_geo_columns([_payload(0, 0)])
        malformed = (
            {"not": [_clause()]},
            {"and": []},
            {"and": {}},
            {"and": [{}]},
            {"and": [{"location": {"range": {"lt": 2}}}]},
            {"and": [_clause(field="missing")]},
        )
        for conditions in malformed:
            with self.subTest(conditions=conditions), self.assertRaises(ValueError):
                _condition_mask(conditions, columns, 1)
        for criteria in (
            {"lat": 0, "lon": 0},
            {"lat": 91, "lon": 0, "radius": 1},
            {"lat": 0, "lon": 181, "radius": 1},
            {"lat": 0, "lon": 0, "radius": -1},
            {"lat": 0, "lon": 0, "radius": float("inf")},
            {"lat": True, "lon": 0, "radius": 1},
        ):
            with self.subTest(criteria=criteria), self.assertRaises(ValueError):
                _geo_mask(*columns["location"], criteria)


class MilvusGeoOverlayTests(unittest.TestCase):
    def test_installer_preserves_original_bases_and_uploader_index_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            directory = workspace / "engine/clients/milvus"
            directory.mkdir(parents=True)
            originals = {
                "configure": (
                    "class MilvusConfigurator:\n"
                    "    def recreate(self, dataset, collection_params):\n"
                    "        return dataset.config.schema\n"
                ),
                "search": (
                    "class MilvusSearcher:\n"
                    "    @classmethod\n"
                    "    def search_one(cls, query, top):\n"
                    "        return cls.collection.search(limit=top)\n"
                ),
                "upload": (
                    "class MilvusUploader:\n"
                    "    @classmethod\n"
                    "    def post_upload(cls, distance):\n"
                    "        cls.collection.create_index(field_name='vector', "
                    "index_params=cls.upload_params)\n"
                ),
            }
            for name, source in originals.items():
                (directory / f"{name}.py").write_text(source, encoding="utf-8")
            before = {name: (directory / f"{name}.py").read_bytes() for name in originals}
            changed = install_milvus_geo_compatibility(workspace)
            self.assertEqual((directory / "upload.py").read_bytes(), before["upload"])
            self.assertFalse(any(path.endswith("/upload.py") for path in changed))
            for name in ("configure", "search"):
                self.assertEqual((directory / f"eigen_{name}_base.py").read_bytes(), before[name])
                self.assertNotEqual((directory / f"{name}.py").read_bytes(), before[name])
            self.assertEqual(
                (directory / "eigen_geo.py").read_bytes(),
                resources.files("eigen").joinpath("geo.py").read_bytes(),
            )
            with self.assertRaises(RunnerError):
                install_milvus_geo_compatibility(workspace)

    def test_installer_rejects_changed_upstream_api_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            directory = workspace / "engine/clients/milvus"
            directory.mkdir(parents=True)
            source = (
                "class MilvusConfigurator:\n"
                "    def recreate(self, incompatible_argument):\n"
                "        pass\n"
            )
            (directory / "configure.py").write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(RunnerError, "unsupported Milvus geo adapter interface"):
                install_milvus_geo_compatibility(workspace)
            self.assertEqual((directory / "configure.py").read_text(encoding="utf-8"), source)
            self.assertEqual([path.name for path in directory.iterdir()], ["configure.py"])

    def test_configurator_strips_only_a_copy_and_keeps_base_configuration(self):
        calls = []

        class BaseConfigurator:
            def recreate(self, dataset, collection_params):
                calls.append((dataset, collection_params))
                return "base-result"

            def execution_params(self, distance, vector_size):
                return {"normalize": distance == "cosine"}

        overlay = _load_overlay("milvus_geo_configure.py.txt", BaseConfigurator)
        schema = {"location": "geo"}
        dataset = SimpleNamespace(config=SimpleNamespace(schema=schema, vector_size=128))
        params = {"sentinel": "preserved"}
        configurator = overlay.MilvusConfigurator()
        self.assertEqual(configurator.recreate(dataset, params), "base-result")
        forwarded, forwarded_params = calls[0]
        self.assertIsNot(forwarded, dataset)
        self.assertIsNot(forwarded.config, dataset.config)
        self.assertEqual(forwarded.config.schema, {})
        self.assertEqual(forwarded.config.vector_size, 128)
        self.assertIs(dataset.config.schema, schema)
        self.assertEqual(schema, {"location": "geo"})
        self.assertIs(forwarded_params, params)
        self.assertEqual(configurator.execution_params("cosine", 128), {"normalize": True})

    def test_configurator_rejects_mixed_or_empty_schema_before_base_call(self):
        base = type("BaseConfigurator", (), {"recreate": Mock()})
        overlay = _load_overlay("milvus_geo_configure.py.txt", base)
        for schema in ({}, {"location": "geo", "tag": "keyword"}):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                overlay.MilvusConfigurator().recreate(
                    SimpleNamespace(config=SimpleNamespace(schema=schema)), {}
                )
        base.recreate.assert_not_called()

    def test_search_keeps_ann_parameters_result_order_and_uses_one_rpc_without_ground_truth(self):
        collection = SimpleNamespace(
            search=Mock(return_value=[SimpleNamespace(ids=[2, 1], distances=[9.0, 1.0])])
        )
        searcher = _searcher(collection)
        with tempfile.TemporaryDirectory() as temporary:
            dataset = _dataset(Path(temporary), [_payload(0, 5), _payload(0, 0), _payload(0, 1)])
            params = {
                "config": {"ef": 83},
                "parallel": 4,
                "eigen_geo": {"dataset_path": str(dataset), "schema": {"location": "geo"}},
            }
            connection = {"port": 19531}
            searcher.init_client("server", "cosine", connection, params)

        class QueryWithoutAccessibleGroundTruth:
            vector = object()
            meta_conditions = {"and": [_clause()]}

            @property
            def expected_result(self):
                raise AssertionError("The ANN adapter must not read ground truth")

        query = QueryWithoutAccessibleGroundTruth()
        result = searcher.search_one(query, 7)
        self.assertEqual(result, [(2, 9.0), (1, 1.0)])
        self.assertEqual(searcher.initialized, [("server", "cosine", connection, params)])
        collection.search.assert_called_once()
        request = collection.search.call_args.kwargs
        self.assertIs(request["data"][0], query.vector)
        self.assertEqual(request["anns_field"], "vector")
        self.assertEqual(request["limit"], 7)
        self.assertEqual(request["param"]["metric_type"], "IP")
        self.assertIs(request["param"]["params"], params["config"])
        self.assertEqual(_expression_ids(request["expr"], 3), [1, 2])

    def test_empty_candidates_still_call_milvus_once(self):
        collection = SimpleNamespace(
            search=Mock(return_value=[SimpleNamespace(ids=[], distances=[])])
        )
        searcher = _searcher(collection)
        searcher.collection, searcher.distance, searcher.search_params = (
            collection,
            "L2",
            {"config": {}},
        )
        searcher.geo_filter = GeoRadiusFilter(_build_geo_columns([_payload(0, 0)]), 1)
        query = SimpleNamespace(vector=[0, 0], meta_conditions={"and": [_clause(radius=0)]})
        self.assertEqual(searcher.search_one(query, 10), [])
        collection.search.assert_called_once()
        self.assertEqual(_expression_ids(collection.search.call_args.kwargs["expr"], 1), [])

    def test_oversized_filter_fails_without_rpc_or_splitting(self):
        collection = SimpleNamespace(search=Mock())
        searcher = _searcher(collection)
        searcher.collection, searcher.distance, searcher.search_params = (
            collection,
            "L2",
            {"config": {}},
        )
        searcher.geo_filter = GeoRadiusFilter(
            _build_geo_columns([_payload(0, 0), _payload(0, 1), _payload(0, 5)]),
            3,
            max_filter_bytes=8,
        )
        with self.assertRaisesRegex(ValueError, "refusing to split ANN search"):
            searcher.search_one(
                SimpleNamespace(vector=[0, 0], meta_conditions={"and": [_clause()]}), 2
            )
        collection.search.assert_not_called()

    def test_missing_settings_and_uninitialized_search_fail_closed(self):
        collection = SimpleNamespace(search=Mock())
        searcher = _searcher(collection)
        with self.assertRaisesRegex(ValueError, "settings"):
            searcher.init_client("server", "l2", {}, {"config": {}})
        self.assertEqual(searcher.initialized, [])
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            searcher.search_one(SimpleNamespace(vector=[0, 0], meta_conditions=None), 10)
        collection.search.assert_not_called()

    def test_measurement_includes_filter_construction_and_rpc(self):
        clock = [0.0]
        events = []

        def make_expression(conditions):
            events.append("filter")
            clock[0] += 0.02
            return "id in [1]"

        def server_search(**kwargs):
            events.append("rpc")
            clock[0] += 0.03
            return [SimpleNamespace(ids=[1], distances=[0.1])]

        collection = SimpleNamespace(search=server_search)
        searcher = _searcher(collection)
        searcher.collection, searcher.distance, searcher.search_params = (
            collection,
            "L2",
            {"config": {}},
        )
        searcher.geo_filter = SimpleNamespace(expression=make_expression)
        namespace = {"time": SimpleNamespace(perf_counter=lambda: clock[0]), "DEFAULT_TOP": 10}
        source = (
            resources.files(_OVERLAY_PACKAGE)
            .joinpath("benchmark_search_one.py.txt")
            .read_text(encoding="utf-8")
        )
        exec(compile(source, "benchmark_search_one.py.txt", "exec"), namespace)
        query = SimpleNamespace(
            vector=[1, 2], expected_result=[1], meta_conditions={"and": [_clause()]}
        )
        recall, elapsed = namespace["_search_one"](searcher, query, 1)
        self.assertEqual(events, ["filter", "rpc"])
        self.assertEqual(recall, 1.0)
        self.assertAlmostEqual(elapsed, 0.05)


if __name__ == "__main__":
    unittest.main()
