"""记忆体检：体量红线 + 归档，别让记忆越写越肥。

来自淋雨家工作台那套「§〇 ≤4 KB / 主文件 ≤25 KB / 超了搬归档留指针」：
- 交接是硬上限：超 4 KB 就报 violation（她开场读不完，等于没写）；
- 其余是告警：事实 200 行、未闭合事件 30、人卡 300、单文件 25 KB、原文 200 MB；
- `fix()` 只做安全动作：先备份，再把「已关闭且过期」的事实搬进
  `_历史\\事实-归档.md`，把「从没被想起、按遗忘曲线已经淡没了」的搬进
  `_历史\\事实-遗忘.md`，事件归档复用 `Memory.archive_events()`。
  仍然成立、又还被想得起的事实一条都不动 —— 体检不该替她删记忆。
"""

from __future__ import annotations

import re
from datetime import date

from . import forget
from .memory import FACT_HEAD, Memory
from .retrieve import IndexDB, parse_loose_date
from .util import append_text, table

RAW_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")

FORGET_POINTER = "<!-- 遗忘：{n} 条（{day}）→ _历史\\事实-遗忘.md -->"


def _closed_before(f, today: date, days: int) -> bool:
    """已关闭、且关闭（或记录）日期早于 days 天前。日期不明的不动。"""
    if not f.closed:
        return False
    ref = f.valid_to if f.valid_to and f.valid_to != "未知" else f.noted
    try:
        d = date.fromisoformat((ref or "")[:10])
    except ValueError:
        return False
    return (today - d).days >= max(0, int(days))


def _hit_stats(mem: Memory) -> dict | None:
    """索引库里的 hit 统计；没有索引库就返回 None（不能替她忘）。"""
    index = IndexDB(mem.cfg)
    if not index.available():
        return None
    return index.hit_stats()


def _recall_of(f, today: date, stats: dict | None, base_days: float) -> float:
    st = (stats or {}).get(f"{f.subject}|{f.topic}") or {}
    base = forget.stability0(base_days, f.importance)
    touched = st.get("last") or parse_loose_date(f.noted, today)
    return forget.retrievability(float(st.get("stability") or 0.0) or base, touched, today)


def _forgettable(mem: Memory, today: date, stats: dict | None = None, facts: list | None = None) -> list:
    """按遗忘曲线该「想不起来」的旧事实。

    四条同时成立才进名单：**一次都没被想起来过**（hit 计数为 0）、够老、
    可回忆度掉到几乎为零、重要度不是「高」。
    没有索引库时一律不动 —— 不知道想起来过没有，就不能替她忘。
    `facts` 传进来时用同一批对象，`fix()` 靠 id 判断哪些要搬走。
    """
    knobs = mem.cfg.knobs
    if not getattr(knobs, "forget_enabled", True):
        return []
    if stats is None:
        stats = _hit_stats(mem)
        if stats is None:
            return []
    below = float(getattr(knobs, "forget_archive_below", 0.02))
    days = int(getattr(knobs, "forget_archive_days", 270))
    base_days = float(getattr(knobs, "forget_stability_days", forget.DEFAULT_STABILITY))
    out = []
    for f in (facts if facts is not None else mem.facts()):
        if f.closed or (f.importance or "").strip() == "高":
            continue
        st = stats.get(f"{f.subject}|{f.topic}") or {}
        if int(st.get("count") or 0) > 0:
            continue  # 想起来过，不算忘
        noted = parse_loose_date(f.noted, today)
        if not noted or (today - noted).days < days:
            continue
        if _recall_of(f, today, stats, base_days) < below:
            out.append(f)
    return out


def curve_summary(mem: Memory, today: date, facts: list | None = None, stats: dict | None = None) -> dict:
    """还成立的事实按「可回忆度」分档（体检 / status 里给人看）。"""
    knobs = mem.cfg.knobs
    base_days = float(getattr(knobs, "forget_stability_days", forget.DEFAULT_STABILITY))
    out = {"记得牢": 0, "记得": 0, "有点淡": 0, "快忘了": 0, "基本忘了": 0}
    for f in (facts if facts is not None else mem.facts()):
        if f.closed:
            continue
        out[forget.label(_recall_of(f, today, stats, base_days))] += 1
    return out


def snapshot(mem: Memory) -> dict:
    cfg = mem.cfg
    knobs = cfg.knobs
    facts = mem.facts()
    events = mem.events()
    today = date.today()
    files = {
        "事实.md": cfg.memory_dir / "事实.md",
        "事件.md": cfg.memory_dir / "事件.md",
        "关系.md": cfg.memory_dir / "关系.md",
        "索引.md": cfg.memory_dir / "索引.md",
        "交接.md": cfg.handover_path,
    }
    sizes = {name: (p.stat().st_size if p.exists() else 0) for name, p in files.items()}
    raw = [p for p in cfg.memory_dir.glob("*.md") if RAW_RE.match(p.name)]
    old_closed = [f for f in facts if _closed_before(f, today, int(knobs.fact_archive_days))]
    stats = _hit_stats(mem)
    return {
        "facts": len(facts),
        "facts_open": len([f for f in facts if not f.closed]),
        "facts_closed": len([f for f in facts if f.closed]),
        "facts_archivable": len(old_closed),
        "facts_forgettable": len(_forgettable(mem, today, stats=stats, facts=facts)) if stats is not None else 0,
        "curve": curve_summary(mem, today, facts=facts, stats=stats),
        "events": len(events),
        "events_open": len([e for e in events if e.status in ("进行中", "等结果")]),
        "people": len(mem.known_people()),
        "handover_bytes": mem.handover_bytes() if cfg.handover_path.exists() else 0,
        "files": sizes,
        "raw_files": len(raw),
        "raw_bytes": sum(p.stat().st_size for p in raw),
        "memory_bytes": sum(p.stat().st_size for p in cfg.memory_dir.rglob("*") if p.is_file()),
        "backups": len(mem.backups()),
        "bad": mem.bad_count(),
    }


def check(mem: Memory) -> list[dict]:
    """返回 findings：level = violation（硬失败）/ warn（告警）。"""
    snap = snapshot(mem)
    knobs = mem.cfg.knobs
    out: list[dict] = []

    def add(level: str, code: str, msg: str, hint: str = "") -> None:
        out.append({"level": level, "code": code, "msg": msg, "hint": hint})

    if not mem.cfg.handover_path.exists():
        add("warn", "handover_missing", "交接.md 还没生成", "等下一次 05:00 日批，或跑 tide batch --force")
    elif snap["handover_bytes"] > int(knobs.handover_max_kb) * 1024:
        add(
            "violation",
            "handover_over",
            f"交接 {snap['handover_bytes']} B 超 {knobs.handover_max_kb} KB 红线",
            "日批会重写；先删掉过期的项目行，别把交接写成周报",
        )

    if snap["facts"] > int(knobs.facts_max_rows):
        add(
            "warn",
            "facts_many",
            f"事实 {snap['facts']} 行超 {knobs.facts_max_rows} 行",
            f"tide memory-check --fix 会把已关闭超过 {knobs.fact_archive_days} 天的搬进 _历史\\事实-归档.md",
        )
    if snap["facts_archivable"]:
        add(
            "warn",
            "facts_archivable",
            f"{snap['facts_archivable']} 条已关闭事实超过 {knobs.fact_archive_days} 天",
            "tide memory-check --fix 归档（会先备份）",
        )
    if snap["facts_forgettable"]:
        add(
            "warn",
            "facts_forgettable",
            f"{snap['facts_forgettable']} 条事实按遗忘曲线已经想不起来了"
            f"（从没被想起、超过 {knobs.forget_archive_days} 天）",
            "tide memory-check --fix 搬进 _历史\\事实-遗忘.md（先备份，能翻回来）",
        )
    if snap["events_open"] > int(knobs.events_open_max):
        add(
            "warn",
            "events_many",
            f"未闭合事件 {snap['events_open']} 条超 {knobs.events_open_max}",
            "把「等结果」超过一个月的关掉，或写进事件备注再结案",
        )
    if snap["people"] > int(knobs.people_max):
        add(
            "warn",
            "people_many",
            f"人卡 {snap['people']} 张超 {knobs.people_max}",
            "按重要度合并/归档；群友的卡只留「群里能说的」",
        )
    for name, size in snap["files"].items():
        if name == "交接.md":
            continue
        if size > int(knobs.memory_file_max_kb) * 1024:
            add(
                "warn",
                "file_big",
                f"{name} {size / 1024:.1f} KB 超 {knobs.memory_file_max_kb} KB",
                "搬旧内容进 _历史\\ 并留指针；别把日志/长输出贴进记忆",
            )
    if snap["raw_bytes"] > int(knobs.raw_total_max_mb) * 1024 * 1024:
        add(
            "warn",
            "raw_big",
            f"原文 {snap['raw_bytes'] / 1024 / 1024:.1f} MB 超 {knobs.raw_total_max_mb} MB",
            "原文只追加、不可重建；按年打包冷备，不要删",
        )
    if snap["bad"]:
        add("warn", "bad_lines", f"坏行 {snap['bad']} 行", "打开 _坏行.md 逐条修；坏行不进检索")
    if not snap["backups"]:
        add("warn", "no_backup", "还没有备份", "跑一次日批，或 tide memory-check --fix")
    return out


def report(mem: Memory) -> dict:
    snap = snapshot(mem)
    findings = check(mem)
    return {
        "ok": not any(f["level"] == "violation" for f in findings),
        "snapshot": snap,
        "findings": findings,
    }


def format_report(rep: dict) -> str:
    s = rep["snapshot"]

    def _size(v: int) -> str:
        return f"{v} B" if v < 1024 else f"{v / 1024:.0f} KB"

    curve = " · ".join(f"{k} {v}" for k, v in (s.get("curve") or {}).items() if v) or "还没有事实"
    lines = [
        "记忆体检：",
        f"  事实 {s['facts']} 行（有效 {s['facts_open']} / 已关闭 {s['facts_closed']}"
        f" / 遗忘候选 {s['facts_forgettable']}）"
        f" · 事件 {s['events']} 条（未闭合 {s['events_open']}）",
        f"  遗忘曲线：{curve}",
        f"  人卡 {s['people']} 张 · 原文 {s['raw_files']} 个 {s['raw_bytes'] / 1024 / 1024:.1f} MB"
        f" · 记忆共 {s['memory_bytes'] / 1024 / 1024:.1f} MB",
        f"  交接 {s['handover_bytes']} B（红线 4096）· 备份 {s['backups']} 份 · 坏行 {s['bad']}",
        f"  单文件：{'、'.join(f'{k} {_size(v)}' for k, v in s['files'].items())}",
    ]
    if not rep["findings"]:
        lines.append("结论：通过（没有越线）")
    else:
        lines.append("结论：" + ("有硬性越线" if not rep["ok"] else f"{len(rep['findings'])} 条告警"))
        for f in rep["findings"]:
            mark = "✗" if f["level"] == "violation" else "⚠"
            lines.append(f"  {mark} {f['msg']}")
            if f["hint"]:
                lines.append(f"    → {f['hint']}")
    return "\n".join(lines)


def fix(mem: Memory) -> dict:
    """安全动作：备份 → 归档已关闭旧事实 / 遗忘曲线淡掉的事实 → 事件归档。"""
    cfg = mem.cfg
    knobs = cfg.knobs
    today = date.today()
    facts = mem.facts()
    old = [f for f in facts if _closed_before(f, today, int(knobs.fact_archive_days))]
    faded = _forgettable(mem, today, facts=facts)
    result = {"backup": "", "facts": 0, "forgotten": 0, "events": 0}
    if old or faded:
        backup = mem.backup("体检")
        result["backup"] = backup.name
        gone = {id(f) for f in old} | {id(f) for f in faded}
        kept = [f for f in facts if id(f) not in gone]
        pointers = []
        if old:
            append_text(
                cfg.history_dir / "事实-归档.md",
                f"\n## {today.isoformat()} 归档（{len(old)} 条已关闭）\n\n"
                + table(FACT_HEAD, [f.as_row() for f in old]),
            )
            pointers.append(
                f"<!-- 归档：{len(old)} 条已关闭事实（{today.isoformat()}）"
                f"→ _历史\\事实-归档.md -->"
            )
            result["facts"] = len(old)
        if faded:
            append_text(
                cfg.history_dir / "事实-遗忘.md",
                f"\n## {today.isoformat()} 遗忘归档（{len(faded)} 条，遗忘曲线）\n\n"
                "<!-- 不是删掉：从没被想起过、可回忆度已低于阈值。"
                "按日期追问（「以前/当时」）仍可翻回来。 -->\n\n"
                + table(FACT_HEAD, [f.as_row() for f in faded]),
            )
            pointers.append(FORGET_POINTER.format(n=len(faded), day=today.isoformat()))
            result["forgotten"] = len(faded)
        mem.write_facts(kept, pointers=pointers)
        mem.audit(
            "体检归档",
            "事实.md",
            f"关闭 {result['facts']} 条 → 事实-归档.md；"
            f"遗忘 {result['forgotten']} 条 → 事实-遗忘.md",
            by="体检",
        )
    try:
        result["events"] = mem.archive_events()
    except Exception as e:  # noqa: BLE001 归档失败不影响已经搬完的事实
        result["events"] = 0
        mem.audit("体检归档", "事件.md", f"失败：{e}", by="体检")
    if result["facts"] or result["forgotten"] or result["events"]:
        mem.audit(
            "体检",
            "记忆",
            f"事实 {result['facts']} 条、遗忘 {result['forgotten']} 条、事件 {result['events']} 条",
            by="体检",
        )
    return result
