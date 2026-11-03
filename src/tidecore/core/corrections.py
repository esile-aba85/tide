"""当天纠正：主人一句「不是 X，是 Y」，立刻改事实，不等第二天 05:00 日批。

只处理「明确指认了新旧两个值」的句子；「我说错了」这种没给新值的不动，
免得把一个模糊的自我否定写成新事实。

写的是同一个 topic：纠正的是值，不是主题。新旧都不删——旧行写有效期止，
以后问「当时」还答得出来；「忘掉」才会真删。
"""

from __future__ import annotations

import re
from datetime import datetime

from . import clock
from .memory import Fact, Memory, shares_object
from .retrieve import IndexDB

# 不是X，是Y / 猫不是，是狗
_PAIR = re.compile(
    r"(?:不是|并不是|不是那个)\s*([^，。！？,；]{1,16})\s*[，,。；;]?\s*(?:而)?是\s*([^，。！？,；]{1,24})"
)
_PAIR_REV = re.compile(r"([^，。！？,；]{1,16})\s*不是[，,]\s*是\s*([^，。！？,；]{1,24})")


def parse(text: str) -> tuple[str, str] | None:
    t = (text or "").strip()
    # 问句不是纠错：「你是不是不喜欢香菜，是过敏吧？」里的「不是」是问法。
    if re.search(r"[？?]", t) or "是不是" in t:
        return None
    m = _PAIR.search(t) or _PAIR_REV.search(t)
    if not m:
        return None
    old = m.group(1).strip(" 了，,。 ")
    new = m.group(2).strip(" 了，,。 ")
    if not old or not new or old == new:
        return None
    return old, new


def apply_correction(mem: Memory, text: str, now: datetime) -> str:
    """把「不是 X，是 Y」落到事实表。返回给人看的说明，没改就返回空串。"""
    pair = parse(text)
    if not pair:
        return ""
    old_text, new_text = pair
    owner = mem.cfg.owner_name
    facts = mem.facts()
    # 只认「旧值真的出现在这条事实里」：以前用 shares_object（任一共同汉字），
    # 「不是蓝色，是红色」会把「常穿黑色外套」也关掉（共用「色」）。
    targets = [
        f
        for f in facts
        if not f.closed
        and f.visibility == "私"
        and f.subject == owner
        and (old_text in f.content or old_text in f.topic)
    ]
    if not targets:
        return ""
    # 新值挂到「正好像这个旧值」的那条上；没有就挂到最近那条。
    target = max(targets, key=lambda f: (f.content.strip() == old_text, f.noted or ""))
    day = clock.logical_date(now).isoformat()
    for f in targets:
        if f.noted and day < f.noted:
            continue
        f.valid_to = day
    facts.append(
        Fact(
            subject=owner,
            topic=target.topic,
            content=new_text,
            speaker=owner,
            trust="本人",
            source="明说",
            confidence=0.92,
            visibility="私",
            importance=target.importance if target.importance not in ("", "—") else "中",
            noted=day,
            origin=f"{now:%m-%d} 纠错：{text[:28]}",
        )
    )
    mem.write_facts(facts)
    mem.audit("纠错", target=target.topic, detail=f"{old_text}→{new_text}", by="主人")
    try:
        IndexDB(mem.cfg).rebuild(mem)
    except Exception:  # noqa: BLE001 索引坏了不能拖垮这一轮
        pass
    return f"{old_text} → {new_text}"
