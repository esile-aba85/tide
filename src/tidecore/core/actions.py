"""她回复里的「隐藏命令」：<cmd>{"type":"draw",...}</cmd>。

LLM 照常聊天，判断出他要画/唱时在回复末尾加一行命令；系统先把这行截掉，
人话照发，命令再后台执行 —— 这样既有活人感，又不靠正则硬猜。
"""

from __future__ import annotations

import json
import re

_CMD = re.compile(r"<cmd>\s*(\{.*?\})\s*</cmd>", re.S | re.I)
_TYPES = ("draw", "sing", "body")


def body_list(act: dict) -> list[str]:
    """把 body 命令里的动作名拍平成数组：do 可以是 "nod"，也可以是 ["nod","shy"]。"""
    raw = act.get("do") or act.get("actions") or act.get("gesture") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out = [str(x).strip() for x in raw if str(x).strip()]
    return out[:8]        # 一条回复最多 8 个动作，别让她抖成筛子


def extract(text: str, types: tuple[str, ...] | list[str] | None = None) -> tuple[str, list[dict]]:
    """返回 (去掉命令后的文本, 命令列表)。解析不了的命令行直接丢掉。

    types：这一版允许哪些动作类型。默认 draw/sing；扩展可以用
    `registry.action_defs()` 把自定义类型加进来。
    """
    allowed = tuple(types or _TYPES)
    raw = text or ""
    actions: list[dict] = []
    for m in _CMD.finditer(raw):
        try:
            data = json.loads(m.group(1))
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict) or data.get("type") not in allowed:
            continue
        actions.append(data)
    clean = _CMD.sub("", raw)
    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, actions
