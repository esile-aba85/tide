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
from datetime import date, datetime
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

DEFAULT_OWNER = {
    "owner_id": "owner",
    "owner_name": "You",
    "owner_display": "you",
    "admins": [],
    "allow_stranger_private": False,
    "groups": {},
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
    owner = json.dumps(DEFAULT_OWNER, ensure_ascii=False, indent=2) + "\n"
    if _write_if_missing(root / "config" / "owner.json", owner):
        created.append("config/owner.json")
    llm = json.dumps(DEFAULT_CONFIG["llm"], ensure_ascii=False, indent=2) + "\n"
    if _write_if_missing(root / "config" / "llm.json", llm):
        created.append("config/llm.json")

    print(f"initialised: {root}")
    if created:
        for name in created:
            print(f"  created {name}")
    else:
        print("  already initialised (nothing overwritten)")
    print("\nnext:")
    print(f"  tide doctor --dir {root}")
    print(f"  tide chat   --dir {root} --offline")
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


def cmd_memory_check(args: argparse.Namespace) -> int:
    from .core.config import Config
    from .core.hygiene import check
    from .core.service import Service

    root = _root(args)
    if not (root / "config" / "tide.json").is_file():
        print(f"not initialised: {root}")
        print(f"run: tide init --dir {root}")
        return 1

    service = Service(Config.load(root, offline=True))
    if getattr(args, "fix", False):
        from .core.hygiene import fix

        result = fix(service.mem)
        print(
            "fixed: facts {facts}, forgotten {forgotten}, events {events}, backup {backup}".format(
                **result
            )
        )
    findings = check(service.mem)
    if not findings:
        print("memory-check: ok")
        return 0

    violations = 0
    for finding in findings:
        level = str(finding.get("level") or "warn")
        if level == "violation":
            violations += 1
        print(f"[{level}] {finding.get('msg')}")
        hint = str(finding.get("hint") or "").strip()
        if hint:
            print(f"  hint: {hint}")
    print(f"{len(findings)} finding(s), {violations} violation(s)")
    return 1 if violations else 0


def cmd_rollback(args: argparse.Namespace) -> int:
    from .core.config import Config
    from .core.service import Service

    root = _root(args)
    if not (root / "config" / "tide.json").is_file():
        print(f"not initialised: {root}")
        print(f"run: tide init --dir {root}")
        return 1

    service = Service(Config.load(root, offline=True))
    message = service.rollback(args.name)
    print(message)
    return 1 if any(word in message for word in ("失败", "没有", "不合法", "找不到")) else 0


def cmd_chat(args: argparse.Namespace) -> int:
    from .core.config import Config
    from .core.service import Service

    root = _root(args)
    if not (root / "config" / "tide.json").is_file():
        print(f"not initialised: {root}")
        print(f"run: tide init --dir {root}")
        return 1

    cfg = Config.load(root, offline=args.offline)
    service = Service(cfg, offline=args.offline)
    owner = cfg.owner_id or "owner"

    print(f"tide chat — data: {root}")
    print(f"model: {'offline' if args.offline else cfg.llm().model}")
    print("type 'exit' or Ctrl-D to leave\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break
        result = service.turn(
            owner,
            text,
            session_type="private",
            channel="cli",
            now=datetime.now(),
            offline=args.offline,
        )
        if result.reply:
            for part in result.parts or [result.reply]:
                print(f"her> {part}")
            print()
        else:
            print(f"her> (no reply: {result.error or 'ignored'})\n")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from .core.config import Config
    from .core.service import Service

    root = _root(args)
    if not (root / "config" / "tide.json").is_file():
        print(f"not initialised: {root}")
        print(f"run: tide init --dir {root}")
        return 1

    cfg = Config.load(root, offline=args.rules)
    service = Service(cfg, offline=args.rules)
    target = date.fromisoformat(args.date) if args.date else None
    result = service.consolidate(target=target, force=args.force, rules=args.rules)

    print(result.message)
    for error in result.errors:
        print(f"  error: {error}")
    print(f"backup: {result.backup or '-'}")
    return 0 if result.ok else 1


def cmd_demo(args: argparse.Namespace) -> int:
    from .core.demo import run_demo

    passed = run_demo(_root(args))
    return 0 if passed >= 20 else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from .adapters.http import run
    from .core.config import Config
    from .core.service import Service

    root = _root(args)
    if not (root / "config" / "tide.json").is_file():
        print(f"not initialised: {root}")
        print(f"run: tide init --dir {root}")
        return 1

    cfg = Config.load(root, offline=args.offline)
    service = Service(cfg, offline=args.offline)
    run(service, host=args.host, port=args.port)
    return 0


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
        ("memory-check", cmd_memory_check, "check memory red lines"),
        ("rollback", cmd_rollback, "restore a daily-batch snapshot"),
        ("chat", cmd_chat, "talk to her in the terminal"),
        ("batch", cmd_batch, "run the daily consolidation for a day"),
        ("demo", cmd_demo, "run the synthetic three-day demo"),
        ("serve", cmd_serve, "run the HTTP bridge"),
    ]
    for name, func, help_text in specs:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--dir", default="./her", help="companion data directory")
        if name in {"chat", "serve"}:
            p.add_argument("--offline", action="store_true", help="do not call a model")
        if name == "batch":
            p.add_argument("--date", default=None, help="logical day, YYYY-MM-DD")
            p.add_argument("--force", action="store_true", help="re-run even if up to date")
            p.add_argument("--rules", action="store_true", help="no model; rule-based only")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1", help="bind address")
            p.add_argument("--port", type=int, default=8787, help="bind port")
        if name == "rollback":
            p.add_argument("--name", default=None, help="snapshot name; newest if omitted")
        if name == "memory-check":
            p.add_argument("--fix", action="store_true", help="archive old facts/events (with backup)")
        p.set_defaults(func=func)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to the legacy code page; memory is UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return int(func(args))
