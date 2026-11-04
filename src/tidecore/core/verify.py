"""20 题回归集：检查「装配进提示词的内容」，不是模型最终那句话。

模型的话不可复现；装配层可复现，也正好是记忆系统真正负责的部分。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import Config
from .service import Service
from .util import parse_table, read_text


def run_verify(
    root: str | Path,
    verbose: bool = True,
    now: datetime | None = None,
) -> tuple[int, int]:
    """跑回归集。

    now 固定住「今天」：种子数据是某几天的，不固定就每年都要改题。
    `tide demo`
    """
    cfg = Config.load(root)
    service = Service(cfg, offline=True)
    path = cfg.acceptance_dir / "回归集.md"
    rows, _ = parse_table(read_text(path), expected=6)
    rows = [r for r in rows if str(r.get("#", "")).strip().isdigit()]
    if not rows:
        if verbose:
            print(f"没有找到回归集：{path}")
        return 0, 0

    when = now or datetime.now()
    passed = 0
    failures = []
    for r in rows:
        scenario = (r.get("场景") or "private").strip()
        question = (r.get("问法") or "").strip()
        expect = [x.strip() for x in (r.get("期望要点") or "").replace("、", "|").split("|") if x.strip()]
        forbid = [x.strip() for x in (r.get("不应出现") or "").replace("、", "|").split("|") if x.strip()]
        group = scenario == "group"
        sender = "20001" if group else cfg.owner_id
        nickname = "小王" if group else cfg.owner_name
        asm = service.recall(
            question,
            sender_id=sender,
            session_type="group" if group else "private",
            group_id="90001" if group else None,
            nickname=nickname,
            now=when,
        )
        text = asm.memory_text
        missing = [e for e in expect if e not in text]
        leaked = [f for f in forbid if f in text]
        ok = not missing and not leaked
        if ok:
            passed += 1
        else:
            failures.append((r, missing, leaked, text))
        if verbose:
            mark = "✓" if ok else "✗"
            extra = ""
            if missing:
                extra += f" 缺：{'、'.join(missing)}"
            if leaked:
                extra += f" 漏：{'、'.join(leaked)}"
            print(f"{mark} {r.get('#','?'):>2}. [{scenario}] {question}{extra}")

    if verbose and failures:
        print("\n失败详情：")
        for r, missing, leaked, text in failures[:3]:
            print(f"\n#{r.get('#')} {r.get('问法')}")
            if missing:
                print("  缺少：" + "、".join(missing))
            if leaked:
                print("  不应出现：" + "、".join(leaked))
            print("  装配内容：")
            for line in text.splitlines():
                print("   " + line)
    if verbose:
        print(f"\n回归集：{passed}/{len(rows)} 通过")
    return passed, len(rows)
