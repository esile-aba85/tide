"""Minimal HTTP adapter for the Tide core.

One request at a time: ``HTTPServer`` (not ``ThreadingHTTPServer``) because the
core's sessions and memory writer are not thread-safe. Put a queue in front if
you need concurrency.

    GET  /health        liveness, no auth
    GET  /status        memory status text, requires token if configured
    POST /say           {"sender_id": "...", "text": "...", ...}
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "tidecore"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
        pass

    @property
    def service(self):
        return self.server.service  # type: ignore[attr-defined]

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = self.server.token  # type: ignore[attr-defined]
        if not token:
            return True
        got = self.headers.get("X-Bridge-Token") or ""
        if not got:
            auth = self.headers.get("Authorization") or ""
            if auth.startswith("Bearer "):
                got = auth[7:].strip()
        return got == token

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            cfg = self.service.cfg
            self._send(
                200,
                {
                    "ok": True,
                    "service": "tide-http",
                    "model": cfg.llm().model,
                    "memory": str(cfg.memory_dir),
                    "pid": os.getpid(),
                },
            )
            return
        if path == "/status":
            if not self._authorized():
                self._send(401, {"ok": False, "error": "bad token"})
                return
            self._send(200, {"ok": True, "text": self.service.status_text()})
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/say":
            self._send(404, {"ok": False, "error": "not found"})
            return
        if not self._authorized():
            self._send(401, {"ok": False, "error": "bad token"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 1_000_000:
            self._send(400, {"ok": False, "error": "empty or oversized body"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"ok": False, "error": "invalid json"})
            return
        if not isinstance(payload, dict):
            self._send(400, {"ok": False, "error": "object expected"})
            return

        sender_id = str(payload.get("sender_id") or "").strip()
        text = str(payload.get("text") or "").strip()
        if not sender_id or not text:
            self._send(400, {"ok": False, "error": "sender_id and text are required"})
            return

        group_id = str(payload.get("group_id") or "").strip() or None
        session_type = "group" if group_id else "private"
        result = self.service.handle(
            sender_id,
            text,
            session_type=session_type,
            group_id=group_id,
            nickname=str(payload.get("nickname") or ""),
            channel="web",
            now=datetime.now(),
            offline=payload.get("offline"),
        )
        self._send(
            200,
            {
                "ok": True,
                "reply": result.reply,
                "parts": result.parts or ([result.reply] if result.reply else []),
                "role": result.role,
                "refused": result.refused,
                "ignored": result.ignored,
                "offline": result.offline,
                "error": result.error,
            },
        )


class Server(HTTPServer):
    allow_reuse_address = os.name != "nt"

    def __init__(self, address: tuple[str, int], service, token: str = ""):
        super().__init__(address, _Handler)
        self.service = service
        self.token = token


def run(service, host: str = "127.0.0.1", port: int = 8787, token: str = "") -> None:
    token = token or os.environ.get("TIDE_BRIDGE_TOKEN", "")
    if host not in LOOPBACK and not token:
        raise SystemExit(
            "refusing to bind a non-loopback address without TIDE_BRIDGE_TOKEN"
        )
    server = Server((host, port), service, token)
    print(f"tide bridge: http://{host}:{server.server_port} (token {'on' if token else 'off'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
