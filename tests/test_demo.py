from __future__ import annotations

import contextlib
import io
import shutil
import unittest
from pathlib import Path

from tidecore.core import demo

BASE = Path(__file__).resolve().parent / "_tmp"


class TestDemo(unittest.TestCase):
    def test_three_day_demo_passes_regression(self) -> None:
        root = BASE / "demo_case"
        shutil.rmtree(root, ignore_errors=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                passed = demo.run_demo(root)
            self.assertEqual(passed, 20)
            self.assertTrue((root / "data" / "memory" / "事实.md").is_file())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
