from __future__ import annotations

import json
import shutil
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from tidecore import cli
from tidecore.adapters.http import Server
from tidecore.core.config import Config
from tidecore.core.service import Service

BASE = Path(__file__).resolve().parent / "_tmp"


class TestHTTPAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.root = BASE / "http"
        shutil.rmtree(self.root, ignore_errors=True)
        cli.main(["init", "--dir", str(self.root)])
        self.service = Service(Config.load(self.root, offline=True), offline=True)
        self.server = Server(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        shutil.rmtree(self.root, ignore_errors=True)

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_health(self) -> None:
        data = self.get("/health")
        self.assertTrue(data["ok"])
        self.assertEqual(data["service"], "tide-http")

    def test_say_round_trip_writes_memory(self) -> None:
        data = self.post(
            "/say",
            {"sender_id": "owner", "text": "我叫阿岚。", "offline": True},
        )
        self.assertTrue(data["ok"], data)
        self.assertIn("reply", data)
        raw = list((self.root / "data" / "memory").glob("2026-*-私聊.md"))
        self.assertEqual(len(raw), 1)
        self.assertIn("阿岚", raw[0].read_text(encoding="utf-8-sig"))

    def test_say_requires_text(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/say", {"sender_id": "owner", "text": ""})
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
