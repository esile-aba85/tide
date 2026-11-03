"""模型用量账本（电表）。

每调一次模型就把返回里的 usage 记一行，所以「缓存命中多少」不用再靠控制台猜。
只记事实（命中 / 未命中 / 输出多少 token），钱是按下面的价目估的。

几条原则：
- 记账失败不能影响回话：写不进去就静默跳过（只读盘、磁盘满、权限问题都算）。
- 一行一个 JSON，追加写；两个进程（本体 / QQ 桥）同时写也不会互相覆盖整行。
- 记的是 token，不是钱；价目改了只改这三个常数，历史账照样能重算。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

# 官方价目（元 / 百万 token），分峰谷两档：
#   缓存命中 0.02 / 0.04    缓存未命中 1 / 2    输出 4 / 8
# 空闲时段按北京时间 00:30–08:30，其余算高峰。官方改了时段就改 _is_idle。
PRICE = {
    "idle": (0.02, 1.0, 4.0),
    "peak": (0.04, 2.0, 8.0),
}

FILE = Path("data") / "llm-usage.jsonl"

_UNITS = (("M", 1_000_000), ("k", 1_000))


def path(root) -> Path:
    return Path(root) / FILE


def pick(usage: dict) -> dict:
    """把各家不一样的叫法归成 hit / miss / out。

    DeepSeek 直接给 prompt_cache_hit_tokens / prompt_cache_miss_tokens；
    OpenAI 那套给 prompt_tokens_details.cached_tokens，命中要从总数里减出来。
    """
    u = usage or {}
    hit = u.get("prompt_cache_hit_tokens")
    miss = u.get("prompt_cache_miss_tokens")
    if hit is None and miss is None:
        detail = u.get("prompt_tokens_details") or {}
        cached = detail.get("cached_tokens")
        total_in = u.get("prompt_tokens")
        if cached is not None and total_in is not None:
            hit = int(cached)
            miss = max(0, int(total_in) - int(cached))
    return {
        "hit": int(hit or 0),
        "miss": int(miss or 0),
        "out": int(u.get("completion_tokens") or 0),
        "total": int(u.get("total_tokens") or 0),
    }


def record(root, model: str, tag: str, usage: dict, ms: float | None = None, now: datetime | None = None) -> None:
    """记一行。usage 是模型返回里的那个 usage 字典。"""
    row = pick(usage)
    if not (row["hit"] or row["miss"] or row["out"]):
        # 有些代理不回 usage：不记空账，免得把命中率算歪
        return
    row.update(
        {
            "at": (now or datetime.now()).isoformat(timespec="seconds"),
            "model": str(model or ""),
            "tag": str(tag or "chat"),
            "ms": int(ms or 0),
        }
    )
    try:
        p = path(root)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def rows(root, days: int = 1, now: datetime | None = None) -> list[dict]:
    """最近 N 天（按自然日，含今天）的账。坏行跳过，不让一行脏数据毁掉整张表。"""
    now = now or datetime.now()
    since = (now - timedelta(days=max(1, days) - 1)).date()
    out: list[dict] = []
    try:
        text = path(root).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            when = datetime.fromisoformat(str(row.get("at")))
        except (ValueError, TypeError):
            continue
        if when.date() >= since:
            out.append(row)
    return out


def _is_idle(when: datetime) -> bool:
    """北京时间的空闲时段：00:30–08:30，输入输出都打对折。"""
    minutes = when.hour * 60 + when.minute
    return 30 <= minutes < 8 * 60 + 30


def price_at(when: datetime | None = None) -> tuple[float, float, float]:
    return PRICE["idle" if _is_idle(when or datetime.now()) else "peak"]


def cost(hit: int, miss: int, out: int, when: datetime | None = None) -> float:
    """一次调用的钱。不传时间就按高峰算（宁可估高，不要让人觉得便宜）。"""
    p_hit, p_miss, p_out = price_at(when)
    return (hit * p_hit + miss * p_miss + out * p_out) / 1_000_000


def _scale(n: int) -> str:
    for unit, size in _UNITS:
        if n >= size:
            return f"{n / size:.2f}{unit}"
    return str(n)


def _money(value: float) -> str:
    """小钱也给个位数：两三次调用就是几厘，写 0.00 会让人以为没记上。"""
    return f"{value:.3f}" if value < 0.1 else f"{value:.2f}"


def summarize(root, days: int = 1, now: datetime | None = None) -> dict:
    got = rows(root, days=days, now=now)
    total = {"n": len(got), "hit": 0, "miss": 0, "out": 0, "ms": 0, "cost": 0.0, "idle": 0, "peak": 0}
    by_tag: dict[str, dict] = {}
    for row in got:
        tag = str(row.get("tag") or "chat")
        bucket = by_tag.setdefault(tag, {"n": 0, "hit": 0, "miss": 0, "out": 0, "cost": 0.0})
        hit, miss, out = (int(row.get(k) or 0) for k in ("hit", "miss", "out"))
        try:
            when = datetime.fromisoformat(str(row.get("at")))
        except (TypeError, ValueError):
            when = now or datetime.now()
        one = cost(hit, miss, out, when)
        for key, value in (("hit", hit), ("miss", miss), ("out", out)):
            total[key] += value
            bucket[key] += value
        total["ms"] += int(row.get("ms") or 0)
        total["cost"] += one
        bucket["cost"] += one
        bucket["n"] += 1
        if _is_idle(when):
            total["idle"] += 1
        else:
            total["peak"] += 1
    return {"total": total, "by_tag": by_tag, "days": max(1, days)}


def _pct(hit: int, miss: int) -> str:
    asked = hit + miss
    return f"{hit / asked:.0%}" if asked else "—"


def report(root, days: int = 1, now: datetime | None = None) -> str:
    data = summarize(root, days=days, now=now)
    t = data["total"]
    if not t["n"]:
        return (
            f"最近 {data['days']} 天没有账。\n"
            "（要么还没重启过，要么模型没回 usage。重启本体后聊几句再看。）"
        )
    lines = [
        f"最近 {data['days']} 天：{t['n']} 次调用，约 {_money(t['cost'])} 元",
        f"  输入  命中 {_scale(t['hit'])} / 未命中 {_scale(t['miss'])}  "
        f"→ 命中率 {_pct(t['hit'], t['miss'])}",
        f"  输出  {_scale(t['out'])}",
        f"  峰谷  空闲 {t['idle']} 次 / 高峰 {t['peak']} 次（00:30–08:30 是半价）",
        f"  平均  {t['ms'] / max(1, t['n']):.0f} ms/次",
    ]
    if len(data["by_tag"]) > 1:
        lines.append("  —— 按用途 ——")
        for tag, b in sorted(data["by_tag"].items(), key=lambda kv: -kv[1]["cost"]):
            lines.append(
                f"  {tag:<10} {b['n']:>4} 次  命中 {_pct(b['hit'], b['miss']):>4}  "
                f"{_scale(b['hit'] + b['miss'] + b['out']):>8} token  {_money(b['cost'])} 元"
            )
    return "\n".join(lines)
