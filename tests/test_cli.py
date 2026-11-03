from __future__ import annotations

import contextlib
import io
import shutil
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from tidecore import cli
from tidecore.core.clock import logical_date

BASE = Path(__file__).resolve().parent / "_tmp"


class TestScaffold(unittest.TestCase):
    def setUp(self) -> None:
        self.root = BASE / self._testMethodName
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def run_cli(self, *argv: str) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(list(argv))
        return code, buf.getvalue()

    def test_version(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            cli.main(["--version"])
        self.assertEqual(caught.exception.code, 0)

    def test_init_creates_layout(self) -> None:
        code, out = self.run_cli("init", "--dir", str(self.root))
        self.assertEqual(code, 0, out)
        self.assertTrue((self.root / "config" / "tide.json").is_file())
        self.assertTrue((self.root / "config" / "persona.md").is_file())
        self.assertTrue((self.root / "data" / "memory").is_dir())
        self.assertTrue((self.root / "data" / "state").is_dir())

    def test_init_does_not_overwrite(self) -> None:
        self.run_cli("init", "--dir", str(self.root))
        persona = self.root / "config" / "persona.md"
        persona.write_text("custom", encoding="utf-8")
        self.run_cli("init", "--dir", str(self.root))
        self.assertEqual(persona.read_text(encoding="utf-8"), "custom")

    def test_doctor_after_init(self) -> None:
        self.run_cli("init", "--dir", str(self.root))
        code, out = self.run_cli("doctor", "--dir", str(self.root))
        self.assertEqual(code, 0, out)
        self.assertIn("all checks passed", out)

    def test_status_empty(self) -> None:
        self.run_cli("init", "--dir", str(self.root))
        code, out = self.run_cli("status", "--dir", str(self.root))
        self.assertEqual(code, 0, out)
        self.assertIn("raw:", out)
        self.assertIn("facts:", out)

    def test_chat_writes_utf8_raw(self) -> None:
        self.run_cli("init", "--dir", str(self.root))
        with mock.patch("builtins.input", side_effect=["我叫阿岚，在做记忆框架。", "exit"]):
            code, out = self.run_cli("chat", "--dir", str(self.root), "--offline")
        self.assertEqual(code, 0, out)
        self.assertIn("her>", out)
        raw = list((self.root / "data" / "memory").glob("2026-*-私聊.md"))
        self.assertEqual(len(raw), 1)
        text = raw[0].read_text(encoding="utf-8-sig")
        self.assertIn("阿岚", text)
        self.assertIn("[cli]", text)

    def test_batch_rules_then_status(self) -> None:
        self.run_cli("init", "--dir", str(self.root))
        with mock.patch("builtins.input", side_effect=["我叫阿岚。", "exit"]):
            self.run_cli("chat", "--dir", str(self.root), "--offline")
        day = logical_date(datetime.now()).isoformat()
        code, out = self.run_cli(
            "batch", "--dir", str(self.root), "--date", day, "--force", "--rules"
        )
        self.assertEqual(code, 0, out)
        self.assertTrue((self.root / "data" / "memory" / "事实.md").is_file())
        code, out = self.run_cli("status", "--dir", str(self.root))
        self.assertEqual(code, 0, out)
        self.assertIn("facts:", out)


if __name__ == "__main__":
    unittest.main()
