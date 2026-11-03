"""Command line interface for Tide.

P0 implements the scaffolding commands only: ``init``, ``doctor`` and
``status``. Commands that need the memory kernel are explicit about not being
available yet instead of silently doing nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import __version__

PERSONA_TEMPLATE = """# Persona

You are a companion who lives on this machine.
Write in short, natural sentences. Remember what the user tells you, and say
when you do not remember instead of inventing details.

## Boundaries

- Never invent facts about the user.
- Private conversations stay private.
"""

DEFAULT_CONFIG = {
    "version": 1,
    "persona": "config/persona.md",
    "memory": "data/memory",
    "state": "data/state",
    "llm": {
        "provider": "openai-compatible",
        "api_base": "https://api.openai.com/v1",
        "model": "",
        "api_key_env": "TIDE_API_KEY",
    },
    "batch": {"hour": 5, "retry_minutes": 10},
    "privacy": {"allow_stranger_private": False},
}

RAW_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")


def _root(args: argparse.Namespace) -> Path:
    return Path(args.dir).expanduser().resolve()


def _write_if_missing(path: Path, text: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def cmd_init(args: argparse.Namespace) -> int:
    root = _root(args)
    for rel in ("config", "data/memory", "data/state"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    if _write_if_missing(root / "config" / "persona.md", PERSONA_TEMPLATE):
        created.append("config/persona.md")
    config = json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2) + "\n"
    if _write_if_missing(root / "config" / "tide.json", config):
        created.append("config/tide.json")

    print(f"initialised: {root}")
    if created:
        for name in created:
            print(f"  created {name}")
    else:
        print("  already initialised (nothing overwritten)")
    print("\nnext:")
    print(f"  tide doctor --dir {root}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    root = _root(args)
    problems = 0

    def report(name: str, ok: bool, detail: str = "") -> None:
        nonlocal problems
        mark = "ok  " if ok else "FAIL"
        line = f"[{mark}] {name}"
        if detail:
            line += f" — {detail}"
        print(line)
        if not ok:
            problems += 1

    report("python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    report("data directory exists", root.exists(), str(root))

    if root.exists():
        probe = root / ".tide-write-probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            writable, detail = True, ""
        except OSError as exc:
            writable, detail = False, f"{type(exc).__name__}: {exc}"
        report("data directory is writable", writable, detail)
    else:
        report("data directory is writable", False, "run `tide init` first")

    report("config/tide.json", (root / "config" / "tide.json").is_file())
    report("config/persona.md", (root / "config" / "persona.md").is_file())
    report("data/memory", (root / "data" / "memory").is_dir())
    report("data/state", (root / "data" / "state").is_dir())

    print()
    if problems:
        print(f"{problems} check(s) failed")
        return 1
    print("all checks passed")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    root = _root(args)
    memory = root / "data" / "memory"
    if not memory.is_dir():
        print(f"not initialised: {memory}")
        print(f"run: tide init --dir {root}")
        return 1

    raw = sorted(p.name for p in memory.glob("*.md") if RAW_RE.match(p.name))
    facts = memory / "事实.md"
    fact_lines = 0
    if facts.is_file():
        for line in facts.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if line.startswith("|") and not re.match(r"^\|[\s:|-]+\|?$", line):
                fact_lines += 1
        fact_lines = max(0, fact_lines - 1)  # header row

    events = memory / "事件.md"
    event_lines = 0
    if events.is_file():
        event_lines = sum(
            1
            for line in events.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            if line.startswith("## ")
        )

    print(f"root:   {root}")
    print(f"memory: {memory}")
    print(f"raw:    {len(raw)} day file(s)")
    print(f"facts:  {fact_lines} row(s)")
    print(f"events: {event_lines} event(s)")
    print(f"index:  {'present' if (memory / '_索引.db').is_file() else 'missing (rebuildable)'}")
    return 0


def cmd_not_yet(args: argparse.Namespace) -> int:
    print(f"`{args.command}` needs the memory kernel, which is not extracted yet.")
    print("P0 ships the package skeleton, `init`, `doctor` and `status`.")
    print("See https://github.com/your-org/tide for the extraction roadmap.")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tide",
        description="Local-first, file-based long-term memory for companion agents.",
    )
    parser.add_argument("--version", action="version", version=f"tidecore {__version__}")
    sub = parser.add_subparsers(dest="command")

    specs = [
        ("init", cmd_init, "create the memory skeleton and config"),
        ("doctor", cmd_doctor, "check the local layout"),
        ("status", cmd_status, "show what is in the memory directory"),
    ]
    for name, func, help_text in specs:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--dir", default="./her", help="companion data directory")
        p.set_defaults(func=func)

    pending = [
        ("chat", "talk to her in the terminal"),
        ("serve", "run the HTTP bridge"),
        ("batch", "run the daily consolidation for a day"),
        ("demo", "run the synthetic three-day demo"),
    ]
    for name, help_text in pending:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--dir", default="./her", help="companion data directory")
        if name == "serve":
            p.add_argument("--port", type=int, default=8787)
        if name == "batch":
            p.add_argument("--date", default=None)
        p.set_defaults(func=cmd_not_yet)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return int(func(args))
