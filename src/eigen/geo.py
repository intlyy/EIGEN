"""Exact spherical payload predicates shared by ground truth and Milvus compatibility.

The Milvus adapter copies this module into its private benchmark snapshot.
It has no dependency on the tuner and never reads vectors or query ground truth
to choose neighbors. Only payload eligibility is computed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EARTH_RADIUS_METERS = 6_371_008.8
MILVUS_GEO_CONTRACT = "payload-id-prefilter-v1"
DEFAULT_MAX_FILTER_BYTES = 8 * 1024 * 1024


def _build_geo_columns(payloads):
    if not payloads:
        return {}
    fields = {
        key
        for payload in payloads
        for key, value in payload.items()
        if isinstance(value, dict) and "lat" in value and "lon" in value
    }
    columns = {}
    for field in sorted(fields):
        try:
            latitude = np.asarray([p[field]["lat"] for p in payloads], dtype=np.float64)
            longitude = np.asarray([p[field]["lon"] for p in payloads], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid geo payload in field {field!r}") from error
        if (
            not np.isfinite(latitude).all()
            or not np.isfinite(longitude).all()
            or np.any(np.abs(latitude) > 90)
            or np.any(np.abs(longitude) > 180)
        ):
            raise ValueError(f"Invalid latitude/longitude in field {field!r}")
        columns[field] = np.radians(latitude), np.radians(longitude)
    return columns


def _geo_mask(latitudes, longitudes, criteria):
    if not isinstance(criteria, dict) or set(criteria) != {"lat", "lon", "radius"}:
        raise ValueError("geo predicates require exactly lat, lon and radius")
    try:
        lat, lon, radius = (float(criteria[key]) for key in ("lat", "lon", "radius"))
    except (TypeError, ValueError) as error:
        raise ValueError("geo center and radius must be numeric") from error
    if (
        not np.isfinite([lat, lon, radius]).all()
        or abs(lat) > 90
        or abs(lon) > 180
        or radius < 0
        or any(isinstance(value, bool) for value in criteria.values())
    ):
        raise ValueError("invalid geo center or radius")
    query_latitude, query_longitude = np.radians([lat, lon])
    latitude_delta = latitudes - query_latitude
    longitude_delta = longitudes - query_longitude
    haversine_a = (
        np.sin(latitude_delta / 2.0) ** 2
        + np.cos(query_latitude) * np.cos(latitudes) * np.sin(longitude_delta / 2.0) ** 2
    )
    haversine_a = np.clip(haversine_a, 0.0, 1.0)
    angular_distance = 2.0 * np.arctan2(np.sqrt(haversine_a), np.sqrt(1.0 - haversine_a))
    return EARTH_RADIUS_METERS * angular_distance < radius


def _condition_mask(conditions, geo_columns, vector_count):
    if conditions is None or conditions == {}:
        return np.ones(vector_count, dtype=bool)
    if not isinstance(conditions, dict) or not set(conditions) <= {"and", "or"}:
        raise ValueError("geo conditions must contain and/or groups")
    result = np.ones(vector_count, dtype=bool)
    for group, entries in conditions.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError("geo condition groups must be nonempty lists")
        masks = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry:
                raise ValueError("geo condition entries must be nonempty objects")
            # Upstream treats each field condition as a separate clause.
            for field, predicate in entry.items():
                if not isinstance(predicate, dict) or set(predicate) != {"geo"}:
                    raise ValueError("this geo workload supports only radius predicates")
                if field not in geo_columns:
                    raise ValueError(f"Geo field {field!r} is missing from payloads")
                masks.append(_geo_mask(*geo_columns[field], predicate["geo"]))
        combine = np.logical_and if group == "and" else np.logical_or
        result &= combine.reduce(masks)
    return result


class GeoRadiusFilter:
    """Calculate eligible row IDs for one timed query; never cache query masks."""

    def __init__(self, columns, count, max_filter_bytes=DEFAULT_MAX_FILTER_BYTES):
        if type(max_filter_bytes) is not int or max_filter_bytes < 1:
            raise ValueError("max_filter_bytes must be a positive integer")
        self.columns, self.count, self.max_filter_bytes = columns, count, max_filter_bytes

    @classmethod
    def from_dataset(cls, dataset_path, schema, max_filter_bytes=DEFAULT_MAX_FILTER_BYTES):
        if not isinstance(schema, dict) or not schema or set(schema.values()) != {"geo"}:
            raise ValueError("Milvus geo compatibility requires an exclusively geo payload schema")
        source = Path(dataset_path)
        vectors = np.load(source / "vectors.npy", mmap_mode="r", allow_pickle=False)
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError("geo dataset requires a nonempty vector matrix")
        count = len(vectors)
        del vectors  # Only the row count is needed; never perform client-side vector search.
        with (source / "payloads.jsonl").open(encoding="utf-8") as handle:
            payloads = [json.loads(line) for line in handle]
        if len(payloads) != count or any(not isinstance(p, dict) for p in payloads):
            raise ValueError("geo payload rows must match vector rows exactly")
        columns = _build_geo_columns(payloads)
        if not set(schema) <= set(columns):
            raise ValueError("declared geo fields are missing from payloads")
        return cls({name: columns[name] for name in schema}, count, max_filter_bytes)

    def expression(self, conditions):
        mask = _condition_mask(conditions, self.columns, self.count)
        ids = np.flatnonzero(mask)
        if len(ids) == 0:
            return "id < 0"  # IDs are zero-based row positions; still execute the server search.
        if len(ids) == self.count:
            return "id >= 0"
        expression = "id in [" + ",".join(map(str, ids.tolist())) + "]"
        if len(expression.encode("ascii")) > self.max_filter_bytes:
            raise ValueError("geo ID filter exceeds max_filter_bytes; refusing to split ANN search")
        return expression
