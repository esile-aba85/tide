from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from tidecore.core.config import Config
from tidecore.core.service import Service

BASE = Path(__file__).resolve().parent / "_tmp"


class TestHooks(unittest.TestCase):
    def setUp(self) -> None:
        self.root = BASE / "hooks"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)
        self.cfg = Config.for_test(self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_before_and_after_are_called(self) -> None:
        service = Service(self.cfg, offline=True)
        calls: list[tuple[str, str]] = []
        service.hooks.before_turn(lambda turn: calls.append(("before", turn.text)))
        service.hooks.after_reply(lambda turn, result: calls.append(("after", result.reply)))
        result = service.turn("10001", "你好", offline=True)
        self.assertTrue(result.reply)
        self.assertEqual(calls[0], ("before", "你好"))
        self.assertEqual(calls[1][0], "after")

    def test_broken_hook_does_not_break_turn(self) -> None:
        service = Service(self.cfg, offline=True)
        reached: list[str] = []

        def boom(turn) -> None:
            raise RuntimeError("plugin bug")

        service.hooks.before_turn(boom)
        service.hooks.before_turn(lambda turn: reached.append("second"))
        result = service.turn("10001", "还在吗", offline=True)
        self.assertTrue(result.reply)
        self.assertEqual(reached, ["second"])


if __name__ == "__main__":
    unittest.main()
