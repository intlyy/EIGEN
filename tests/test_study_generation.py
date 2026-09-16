"""Generated studies retain the requested budget and explicit geo execution contract."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mutune.benchmark_compat import MILVUS_GEO_CONTRACT
from mutune.geo_minidb import build_geo_minidbs
from mutune.study import load_study
from scripts.prepare_study import main


class StudyGenerationTests(unittest.TestCase):
    def _source(self, root):
        source = root / "source"
        source.mkdir()
        np.save(source / "vectors.npy", np.ones((30, 2), dtype=np.float32))
        (source / "payloads.jsonl").write_text(
            "".join(json.dumps({"location": {"lat": i, "lon": 0}}) + "\n" for i in range(30)),
            encoding="utf-8",
        )
        query = {
            "query": [1, 1],
            "conditions": {"and": [{"location": {"geo": {"lat": 0, "lon": 0, "radius": 1}}}]},
            "closest_ids": [0],
        }
        (source / "tests.jsonl").write_text(json.dumps(query) + "\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            build_geo_minidbs(source, root / "minis", top_k=1)
        benchmark = root / "benchmark"
        benchmark.mkdir()
        (benchmark / "run.py").write_text("# fingerprint fixture; never executed\n")
        return root / "minis/manifest.json", benchmark

    def _generate(self, engine, manifest, benchmark, output, *extra):
        arguments = [
            "prepare_study.py",
            "--engine",
            engine,
            "--manifest",
            str(manifest),
            "--benchmark-repo",
            str(benchmark),
            "--output",
            str(output),
            "--top-k",
            "1",
            *extra,
        ]
        with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            main()

    def test_milvus_and_qdrant_geo_studies_use_three_twenty_evaluation_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, benchmark = self._source(root)
            for engine in ("milvus", "qdrant"):
                output = root / engine
                self._generate(engine, manifest, benchmark, output)
                study = load_study(output / "study.json")
                self.assertEqual(len(study.minidbs), 3)
                self.assertEqual([p.config.tuning.budget for p in study.minidbs], [20, 20, 20])
                self.assertEqual(sum(p.config.tuning.budget for p in study.minidbs), 60)
                self.assertEqual(len({p.config.tuning.seed for p in study.minidbs}), 3)
                for project in [*study.minidbs, study.full_database]:
                    self.assertTrue(project.config.execution.filtered)
                    settings = project.config.runner.settings
                    self.assertEqual(settings["dataset_entry"]["schema"], {"location": "geo"})
                    self.assertIn("expected_source_sha256", settings)
                    self.assertEqual(
                        settings.get("milvus_geo_filter"),
                        MILVUS_GEO_CONTRACT if engine == "milvus" else None,
                    )

    def test_milvus_geo_cannot_silently_fall_back_to_unadapted_upstream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, benchmark = self._source(root)
            output = root / "milvus"
            self._generate("milvus", manifest, benchmark, output)
            path = output / "milvus-mini-01.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            del config["runner"]["settings"]["milvus_geo_filter"]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicit payload-ID"):
                load_study(output / "study.json")

    def test_geo_pgvector_is_rejected_and_budget_override_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, benchmark = self._source(root)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self._generate("pgvector", manifest, benchmark, root / "pgvector")
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((root / "pgvector").exists())
            self._generate(
                "milvus", manifest, benchmark, root / "custom", "--budget-per-minidb", "21"
            )
            study = load_study(root / "custom/study.json")
            self.assertEqual([p.config.tuning.budget for p in study.minidbs], [21, 21, 21])


if __name__ == "__main__":
    unittest.main()
