#!/usr/bin/env python3
"""Fail if the repository contains secrets, personal identifiers or host paths.

Run from the repository root:

    python scripts/check_hygiene.py

This is intentionally dumb and fast: it is a release gate, not a scanner.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "build", "dist", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".db"}

PATTERNS = [
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("api key", re.compile(r"\bsk-[A-Za-z0-9]{16,}")),
    ("personal name", re.compile("\u6c5f\u6960")),
    ("persona name", re.compile("\u5927\u80a5\u9c7c")),
    ("owner id", re.compile(r"\b1064959243\b")),
    ("group id", re.compile(r"\b(?:871056101|169746919)\b")),
    ("windows absolute path", re.compile(r"(?:^|[\s\"'(=])[A-Za-z]:\\")),
    ("home absolute path", re.compile(r"/Users/[^/\s\"']+")),
    ("inline secret", re.compile(r'"(?:api_key|token|secret)"\s*:\s*"[^"]{8,}"')),
]


def iter_files() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        out.append(path)
    return out


def main() -> int:
    hits: list[str] = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(ROOT)
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pattern in PATTERNS:
                if pattern.search(line):
                    hits.append(f"{rel}:{lineno}: {name}")

    if not hits:
        print(f"hygiene: clean ({len(iter_files())} files checked)")
        return 0

    print("hygiene: forbidden content found", file=sys.stderr)
    for hit in hits:
        print(f"  {hit}", file=sys.stderr)
    print(f"\n{len(hits)} hit(s). Fix before publishing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
