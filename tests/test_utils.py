from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mutune.utils import atomic_write_json, ensure_within, fingerprint, safe_name


class UtilsTests(unittest.TestCase):
    def test_fingerprint_is_order_independent(self) -> None:
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))

    def test_safe_name(self) -> None:
        self.assertEqual(safe_name("hello / unsafe"), "hello-unsafe")

    def test_atomic_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "value.json"
            atomic_write_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})

    def test_ensure_within_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            with self.assertRaises(ValueError):
                ensure_within(root / ".." / "escape", root)


if __name__ == "__main__":
    unittest.main()
